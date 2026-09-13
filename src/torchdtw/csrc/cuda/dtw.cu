#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>
#include <tuple>
#include <type_traits>
#include <vector>

// Default shared memory budget of 48KiB per block, shared by 3 cost diagonal buffers and
// 3 path-length diagonal buffers, each of length distances.size(3). The buffers are allocated
// dynamically at launch so the per-dtype capacity is 49152 / (3 * (sizeof(acc_t) + sizeof(uint16_t))):
// 1638 for double, 2730 for float/half/bfloat16. The cost buffers hold the accumulator type
// (acc_t<scalar_t>, float for half/bfloat16), not the storage type, so half/bfloat16 share float's
// capacity rather than getting the wider 4096 that 2-byte storage would allow.
#define MAX_SHARED_BYTES 49152

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace torchdtw {

using torch::stable::Tensor;

template <typename T, size_t N, typename index_t>
using PackedTensorAccessor =
    torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, torch::headeronly::RestrictPtrTraits, index_t>;
template <typename T, size_t N> using PackedTensorAccessor32 = PackedTensorAccessor<T, N, int32_t>;
template <typename T, size_t N> using PackedTensorAccessor64 = PackedTensorAccessor<T, N, int64_t>;
template <typename T, size_t N> inline PackedTensorAccessor32<T, N> packed_accessor32(torch::stable::Tensor t) {
  return PackedTensorAccessor32<T, N>(
      static_cast<typename PackedTensorAccessor32<T, N>::PtrType>(t.data_ptr()), t.sizes().data(), t.strides().data());
}
template <typename T, size_t N> inline PackedTensorAccessor64<T, N> packed_accessor64(torch::stable::Tensor t) {
  return PackedTensorAccessor64<T, N>(
      static_cast<typename PackedTensorAccessor64<T, N>::PtrType>(t.data_ptr()), t.sizes().data(), t.strides().data());
}
template <typename T, size_t N, typename index_t>
inline PackedTensorAccessor<T, N, index_t> packed_accessor(torch::stable::Tensor t) {
  return PackedTensorAccessor<T, N, index_t>(
      static_cast<typename PackedTensorAccessor<T, N, index_t>::PtrType>(t.data_ptr()),
      t.sizes().data(),
      t.strides().data());
}

template <typename scalar_t>
using acc_t = std::conditional_t<
    std::is_same_v<scalar_t, torch::headeronly::Half> || std::is_same_v<scalar_t, torch::headeronly::BFloat16>, float,
    scalar_t>;

// Step patterns, as in csrc/dtw.cpp.
enum StepPattern : int32_t { SYMMETRIC1 = 1, SYMMETRIC2 = 2 };

static void check_step_pattern(int64_t step_pattern) {
  STD_TORCH_CHECK(
      step_pattern == SYMMETRIC1 || step_pattern == SYMMETRIC2,
      "step_pattern must be 1 (symmetric1) or 2 (symmetric2)");
}

static cudaStream_t current_stream() {
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  // Available in PyTorch >= 2.13
  // cudaStream_t stream =
  //     static_cast<cudaStream_t>(torch::stable::accelerator::getCurrentStream(device_idx).nativeHandle());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

static int num_threads_for(int64_t max_diag) {
  const int capped_diag = max_diag > 1024 ? 1024 : static_cast<int>(max_diag);
  return ((capped_diag + 31) / 32) * 32 > 1024 ? 1024 : ((capped_diag + 31) / 32) * 32;
}

static void check_launch(const char* kernel) {
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, kernel, " launch failed: ", cudaGetErrorString(err));
}

// Wavefront DP over anti-diagonals. Each cell tracks its cost and the length of the optimal path
// reaching it (one more than its chosen parent's path length). The cost of (N-1, M-1) divided by
// its path length is the final result, so no traceback is needed. With symmetric2 the result is
// normalized by N + M instead, and the path lengths are unused.
template <typename scalar_t, typename sx_t, typename index_t>
__global__ void dtw_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, const PackedTensorAccessor<scalar_t, 4, index_t> distances,
    const PackedTensorAccessor32<sx_t, 1> sx, const PackedTensorAccessor32<sx_t, 1> sy, bool symmetric,
    int32_t step_pattern) {
  int32_t x, y;
  if (symmetric) {
    const int32_t b = static_cast<int32_t>(blockIdx.x);
    y = static_cast<int32_t>((1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(b))) / 2.0f);
    while (static_cast<int64_t>(y) * (y - 1) / 2 > b)
      y--;
    while (static_cast<int64_t>(y + 1) * y / 2 <= b)
      y++;
    x = b - y * (y - 1) / 2;
  } else {
    x = blockIdx.x;
    y = blockIdx.y;
  }
  if (x >= out.size(0) || y >= out.size(1))
    return;
  const int32_t N = static_cast<int32_t>(sx[x]);
  const int32_t M = static_cast<int32_t>(sy[y]);
  if (N <= 0 || M <= 0) {
    if (threadIdx.x == 0) {
      out[x][y] = static_cast<scalar_t>(0);
      if (symmetric)
        out[y][x] = static_cast<scalar_t>(0);
    }
    return;
  }

  // Accumulate cost in acc_t<scalar_t> (float for half/bfloat16) to limit rounding drift.
  using acc_t = torchdtw::acc_t<scalar_t>;
  extern __shared__ unsigned char smem[];
  const int32_t buf_len = static_cast<int32_t>(distances.size(3));
  acc_t* const cost_smem = reinterpret_cast<acc_t*>(smem);
  // Round the cost buffers' size up to 2 bytes so the uint16_t buffers are aligned for 1-byte dtypes.
  const size_t len_offset = (3 * static_cast<size_t>(buf_len) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  uint16_t* const len_smem = reinterpret_cast<uint16_t*>(smem + len_offset);
  // Named pointers rotated by register swaps: a dynamically indexed pointer array would be
  // spilled to local memory and slow down every shared memory access in the wavefront loop.
  acc_t* cost_alpha = cost_smem;               // Last diagonal
  acc_t* cost_beta = cost_smem + buf_len;      // Second to last diagonal
  acc_t* cost_gamma = cost_smem + 2 * buf_len; // Buffer for the next diagonal
  uint16_t* len_alpha = len_smem;
  uint16_t* len_beta = len_smem + buf_len;
  uint16_t* len_gamma = len_smem + 2 * buf_len;

  const auto distances_xy = distances[x][y];

  if (threadIdx.x == 0) {
    cost_gamma[0] = static_cast<acc_t>(distances_xy[0][0]);
    len_gamma[0] = 1;
  }
  __syncthreads();
  acc_t* cost_temp = cost_beta;
  cost_beta = cost_alpha;
  cost_alpha = cost_gamma;
  cost_gamma = cost_temp;
  uint16_t* len_temp = len_beta;
  len_beta = len_alpha;
  len_alpha = len_gamma;
  len_gamma = len_temp;

  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      const acc_t dist = static_cast<acc_t>(distances_xy[i][j]);
      acc_t new_cost;
      uint16_t parent_len;
      // Boundary cells have a single valid parent. No sentinel cost is used for the invalid
      // parents: a real cost can reach the maximum of narrow integer dtypes and tie with it.
      if (i == 0) {
        new_cost = dist + cost_alpha[j - 1];
        parent_len = len_alpha[j - 1];
      } else if (j == 0) {
        new_cost = dist + cost_alpha[j];
        parent_len = len_alpha[j];
      } else if (step_pattern == SYMMETRIC2) {
        // The diagonal step counts the local distance twice, so it is part of the comparison.
        const acc_t c_up = static_cast<acc_t>(cost_alpha[j] + dist);
        const acc_t c_left = static_cast<acc_t>(cost_alpha[j - 1] + dist);
        const acc_t c_diag = static_cast<acc_t>(cost_beta[j - 1] + static_cast<acc_t>(2) * dist);
        if (c_diag <= c_left && c_diag <= c_up) {
          new_cost = c_diag;
          parent_len = len_beta[j - 1];
        } else if (c_left <= c_up) {
          new_cost = c_left;
          parent_len = len_alpha[j - 1];
        } else {
          new_cost = c_up;
          parent_len = len_alpha[j];
        }
      } else {
        const acc_t c_up = cost_alpha[j];
        const acc_t c_left = cost_alpha[j - 1];
        const acc_t c_diag = cost_beta[j - 1];
        acc_t min_cost;
        if (c_diag <= c_left && c_diag <= c_up) {
          min_cost = c_diag;
          parent_len = len_beta[j - 1];
        } else if (c_left <= c_up) {
          min_cost = c_left;
          parent_len = len_alpha[j - 1];
        } else {
          min_cost = c_up;
          parent_len = len_alpha[j];
        }
        new_cost = dist + min_cost;
      }
      cost_gamma[j] = new_cost;
      len_gamma[j] = static_cast<uint16_t>(parent_len + 1);
    }
    __syncthreads();

    acc_t* const cost_temp = cost_beta;
    cost_beta = cost_alpha;
    cost_alpha = cost_gamma;
    cost_gamma = cost_temp;
    uint16_t* const len_temp = len_beta;
    len_beta = len_alpha;
    len_alpha = len_gamma;
    len_gamma = len_temp;
  }

  if (threadIdx.x == 0) {
    const acc_t final_cost = cost_alpha[M - 1];
    const int64_t denominator =
        step_pattern == SYMMETRIC2 ? static_cast<int64_t>(N) + M : static_cast<int64_t>(len_alpha[M - 1]);
    // For integral dtypes divide in int64 (as on CPU): casting the denominator to a
    // narrow dtype can yield 0 (e.g. 256 as int8) and divide by zero.
    scalar_t result;
    if constexpr (std::is_integral_v<scalar_t>) {
      result = static_cast<scalar_t>(static_cast<int64_t>(final_cost) / denominator);
    } else {
      result = static_cast<scalar_t>(final_cost / static_cast<acc_t>(denominator));
    }
    out[x][y] = result;
    if (symmetric)
      out[y][x] = result;
  }
}

template <typename distances_t, typename sx_t>
void dtw_batch_cuda_impl(
    Tensor& out, const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric, int64_t step_pattern) {
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);
  const dim3 num_blocks = symmetric ? dim3(static_cast<unsigned int>(nx * (nx - 1) / 2)) : dim3(nx, ny);
  const int64_t max_diag = max_x < max_y ? max_x : max_y;
  const int num_threads = num_threads_for(max_diag);
  const bool needs_64bit = nx * ny * max_x * max_y > std::numeric_limits<int32_t>::max();
  using acc_t = torchdtw::acc_t<distances_t>;
  const size_t cost_bytes = (3 * static_cast<size_t>(max_y) * sizeof(acc_t) + 1) & ~static_cast<size_t>(1);
  const size_t smem_size = cost_bytes + 3 * static_cast<size_t>(max_y) * sizeof(uint16_t);
  STD_TORCH_CHECK(
      smem_size <= MAX_SHARED_BYTES,
      "distances.size(3) > ",
      MAX_SHARED_BYTES / (3 * (sizeof(acc_t) + sizeof(uint16_t))),
      ": too large to use CUDA shared memory for this dtype");
  const cudaStream_t stream = current_stream();

  if (needs_64bit) {
    dtw_kernel<distances_t, sx_t, int64_t><<<num_blocks, num_threads, smem_size, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor64<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric,
        static_cast<int32_t>(step_pattern));
  } else {
    dtw_kernel<distances_t, sx_t, int32_t><<<num_blocks, num_threads, smem_size, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor32<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric,
        static_cast<int32_t>(step_pattern));
  }
  check_launch("dtw_kernel");
}

Tensor
dtw_batch_cuda(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric, int64_t step_pattern) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");

  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(sx.dim() == 1 && sy.dim() == 1, "sx and sy must be 1D tensors");
  STD_TORCH_CHECK(sx.is_cuda() && sy.is_cuda(), "sx and sy must be on CUDA");
  STD_TORCH_CHECK(
      sx.get_device() == distances.get_device() && sy.get_device() == distances.get_device(),
      "sx and sy must be on the same CUDA device as distances");
  STD_TORCH_CHECK(
      sx.size(0) == nx && sy.size(0) == ny, "sx and sy sizes must match the first two dimensions of distances");
  STD_TORCH_CHECK(!symmetric || nx == ny, "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  check_step_pattern(step_pattern);
  STD_TORCH_CHECK(
      step_pattern == SYMMETRIC2 || max_x + max_y - 1 <= std::numeric_limits<uint16_t>::max(),
      "Sum of sequence lengths exceeds uint16_t path-length capacity");
  STD_TORCH_CHECK(sy.scalar_type() == sx.scalar_type(), "sx and sy dtypes do not match");
  constexpr int64_t max_grid_x = 2147483647; // 2^31 - 1
  constexpr int64_t max_grid_y = 65535;
  if (symmetric) {
    STD_TORCH_CHECK(
        nx * (nx - 1) / 2 <= max_grid_x, "symmetric dtw_batch too large: nx*(nx-1)/2 exceeds the CUDA grid limit");
  } else {
    STD_TORCH_CHECK(nx <= max_grid_x, "distances.size(0) exceeds the CUDA grid limit of 2^31-1");
    STD_TORCH_CHECK(ny <= max_grid_y, "distances.size(1) exceeds the CUDA grid limit of 65535");
  }

  Tensor out =
      symmetric ? torch::stable::new_zeros(distances, {nx, ny}) : torch::stable::new_empty(distances, {nx, ny});
  if (symmetric && nx <= 1)
    return out;
  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_batch_cuda_impl",
      AT_WRAP([&] {
        using distances_t = scalar_t;
        THO_DISPATCH_V2(
            sx.scalar_type(),
            "dtw_batch_cuda_impl_2",
            AT_WRAP([&] {
              using sx_t = scalar_t;
              (dtw_batch_cuda_impl<distances_t, sx_t>(out, distances, sx, sy, symmetric, step_pattern));
            }),
            AT_INTEGRAL_TYPES_V2);
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);
  return out;
}

Tensor dtw_cuda(const Tensor& distances, int64_t step_pattern) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  Tensor sxy = torch::stable::new_empty(distances, {2}, std::make_optional(torch::headeronly::ScalarType::Long));
  Tensor sx = torch::stable::narrow(sxy, 0, 0, 1);
  Tensor sy = torch::stable::narrow(sxy, 0, 1, 1);
  torch::stable::fill_(sx, distances.size(0));
  torch::stable::fill_(sy, distances.size(1));
  Tensor result = dtw_batch_cuda(
      torch::stable::view(distances, {1, 1, distances.size(0), distances.size(1)}), sx, sy, false, step_pattern);
  // Spell out the empty shape: a bare {} selects the view(Tensor, ScalarType) overload in recent PyTorch.
  return torch::stable::view(result, std::vector<int64_t>{});
}

// Wavefront DP over anti-diagonals writing the full N x M accumulated cost matrix to global memory,
// for dtw_traceback_kernel. Same recurrence and tie-breaking as csrc/dtw.cpp.
template <typename scalar_t, typename index_t>
__global__ void dtw_cost_matrix_kernel(
    PackedTensorAccessor<torchdtw::acc_t<scalar_t>, 2, index_t> cost,
    const PackedTensorAccessor<scalar_t, 2, index_t> distances, int32_t step_pattern) {
  using acc_t = torchdtw::acc_t<scalar_t>;
  const int32_t N = static_cast<int32_t>(cost.size(0));
  const int32_t M = static_cast<int32_t>(cost.size(1));

  if (threadIdx.x == 0) {
    cost[0][0] = static_cast<acc_t>(distances[0][0]);
  }
  __syncthreads();

  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      const acc_t dist = static_cast<acc_t>(distances[i][j]);
      if (i == 0) {
        cost[0][j] = dist + cost[0][j - 1];
      } else if (j == 0) {
        cost[i][0] = dist + cost[i - 1][0];
      } else if (step_pattern == SYMMETRIC2) {
        const acc_t c_up = static_cast<acc_t>(cost[i - 1][j] + dist);
        const acc_t c_left = static_cast<acc_t>(cost[i][j - 1] + dist);
        const acc_t c_diag = static_cast<acc_t>(cost[i - 1][j - 1] + static_cast<acc_t>(2) * dist);
        cost[i][j] = (c_diag <= c_left && c_diag <= c_up) ? c_diag : (c_left <= c_up ? c_left : c_up);
      } else {
        const acc_t c_up = cost[i - 1][j];
        const acc_t c_left = cost[i][j - 1];
        const acc_t c_diag = cost[i - 1][j - 1];
        cost[i][j] = dist + ((c_diag <= c_left && c_diag <= c_up) ? c_diag : (c_left <= c_up ? c_left : c_up));
      }
    }
    __syncthreads();
  }
}

// Single-thread traceback from (N-1, M-1) to (0, 0). Writes the path backwards, reverses it in place,
// and outputs the path length and the normalized cost.
template <typename scalar_t, typename index_t>
__global__ void dtw_traceback_kernel(
    int64_t* __restrict__ path, int64_t* __restrict__ path_length, scalar_t* __restrict__ cost_out,
    const PackedTensorAccessor<torchdtw::acc_t<scalar_t>, 2, index_t> cost,
    const PackedTensorAccessor<scalar_t, 2, index_t> distances, int32_t step_pattern) {
  using acc_t = torchdtw::acc_t<scalar_t>;
  const int32_t N = static_cast<int32_t>(cost.size(0));
  const int32_t M = static_cast<int32_t>(cost.size(1));
  int32_t i = N - 1;
  int32_t j = M - 1;
  int32_t len = 0;

  path[0] = i;
  path[1] = j;
  len++;
  while (i > 0 && j > 0) {
    acc_t c_up = cost[i - 1][j];
    acc_t c_left = cost[i][j - 1];
    acc_t c_diag = cost[i - 1][j - 1];
    if (step_pattern == SYMMETRIC2) {
      const acc_t dist = static_cast<acc_t>(distances[i][j]);
      c_up = static_cast<acc_t>(c_up + dist);
      c_left = static_cast<acc_t>(c_left + dist);
      c_diag = static_cast<acc_t>(c_diag + static_cast<acc_t>(2) * dist);
    }
    if (c_diag <= c_left && c_diag <= c_up) {
      i--;
      j--;
    } else if (c_left <= c_up) {
      j--;
    } else {
      i--;
    }
    path[len * 2] = i;
    path[len * 2 + 1] = j;
    len++;
  }
  while (i > 0) {
    i--;
    path[len * 2] = i;
    path[len * 2 + 1] = j;
    len++;
  }
  while (j > 0) {
    j--;
    path[len * 2] = i;
    path[len * 2 + 1] = j;
    len++;
  }

  for (int32_t k = 0; k < len / 2; k++) {
    const int64_t ti = path[k * 2];
    const int64_t tj = path[k * 2 + 1];
    path[k * 2] = path[(len - 1 - k) * 2];
    path[k * 2 + 1] = path[(len - 1 - k) * 2 + 1];
    path[(len - 1 - k) * 2] = ti;
    path[(len - 1 - k) * 2 + 1] = tj;
  }

  *path_length = len;
  const acc_t final_cost = cost[N - 1][M - 1];
  const int64_t denominator = step_pattern == SYMMETRIC2 ? static_cast<int64_t>(N) + M : static_cast<int64_t>(len);
  if constexpr (std::is_integral_v<scalar_t>) {
    *cost_out = static_cast<scalar_t>(static_cast<int64_t>(final_cost) / denominator);
  } else {
    *cost_out = static_cast<scalar_t>(final_cost / static_cast<acc_t>(denominator));
  }
}

template <typename scalar_t, typename index_t>
void dtw_cost_and_path_cuda_impl(
    Tensor& cost_out, Tensor& path, Tensor& path_length, const Tensor& distances, int64_t step_pattern) {
  using acc_t = torchdtw::acc_t<scalar_t>;
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  Tensor cost = torch::stable::new_empty(
      distances, {N, M}, std::make_optional(torch::headeronly::CppTypeToScalarType<acc_t>::value));
  const cudaStream_t stream = current_stream();

  dtw_cost_matrix_kernel<scalar_t, index_t><<<1, num_threads_for(N < M ? N : M), 0, stream>>>(
      packed_accessor<acc_t, 2, index_t>(cost),
      packed_accessor<scalar_t, 2, index_t>(distances),
      static_cast<int32_t>(step_pattern));
  check_launch("dtw_cost_matrix_kernel");

  dtw_traceback_kernel<scalar_t, index_t><<<1, 1, 0, stream>>>(
      reinterpret_cast<int64_t*>(path.data_ptr()),
      reinterpret_cast<int64_t*>(path_length.data_ptr()),
      reinterpret_cast<scalar_t*>(cost_out.data_ptr()),
      packed_accessor<acc_t, 2, index_t>(cost),
      packed_accessor<scalar_t, 2, index_t>(distances),
      static_cast<int32_t>(step_pattern));
  check_launch("dtw_traceback_kernel");
}

std::tuple<Tensor, Tensor> dtw_cost_and_path_cuda(const Tensor& distances, int64_t step_pattern) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  STD_TORCH_CHECK(N + M - 1 <= std::numeric_limits<int32_t>::max(), "Sum of sequence lengths exceeds int32_t");
  check_step_pattern(step_pattern);
  const torch::stable::accelerator::DeviceGuard device_guard(distances.get_device());
  const bool needs_64bit = N * M > std::numeric_limits<int32_t>::max();

  Tensor cost_out = torch::stable::new_empty(distances, {});
  Tensor path =
      torch::stable::new_empty(distances, {N + M - 1, 2}, std::make_optional(torch::headeronly::ScalarType::Long));
  Tensor path_length =
      torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));

  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_cost_and_path_cuda",
      AT_WRAP([&] {
        if (needs_64bit) {
          (dtw_cost_and_path_cuda_impl<scalar_t, int64_t>(cost_out, path, path_length, distances, step_pattern));
        } else {
          (dtw_cost_and_path_cuda_impl<scalar_t, int32_t>(cost_out, path, path_length, distances, step_pattern));
        }
      }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);

  const cudaStream_t stream = current_stream();
  int64_t len;
  cudaMemcpyAsync(&len, path_length.data_ptr(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return {cost_out, torch::stable::narrow(path, 0, 0, len)};
}

Tensor dtw_path_cuda(const Tensor& distances, int64_t step_pattern) {
  return std::get<1>(dtw_cost_and_path_cuda(distances, step_pattern));
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CUDA, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cuda));
  m.impl("dtw_path", &TORCH_BOX(dtw_path_cuda));
  m.impl("dtw_cost_and_path", &TORCH_BOX(dtw_cost_and_path_cuda));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_cuda));
}

} // namespace torchdtw
