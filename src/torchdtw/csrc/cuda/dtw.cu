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

// Shared memory has a size of 48KiB, shared by 3 cost diagonal buffers and 3 path-length diagonal buffers.
// MAX_DIAG_LEN = 49152 / (3 * (sizeof(double) + sizeof(uint16_t))) = 49152 / (3 * (8 + 2))
#define MAX_DIAG_LEN 1638

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

// Wavefront DP over anti-diagonals. Each cell tracks its cost and the length of the optimal path
// reaching it (one more than its chosen parent's path length). The cost of (N-1, M-1) divided by
// its path length is the final result, so no traceback is needed.
template <typename scalar_t, typename sx_t, typename index_t>
__global__ void dtw_kernel(
    PackedTensorAccessor32<scalar_t, 2> out, const PackedTensorAccessor<scalar_t, 4, index_t> distances,
    const PackedTensorAccessor32<sx_t, 1> sx, const PackedTensorAccessor32<sx_t, 1> sy, bool symmetric,
    int step_pattern) {
  int32_t x, y;
  if (symmetric) {
    const int32_t b = static_cast<int32_t>(blockIdx.x);
    y = static_cast<int32_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(b))) / 2.0);
    x = b - y * (y - 1) / 2;
  } else {
    x = blockIdx.x;
    y = blockIdx.y;
  }
  if (x >= out.size(0) || y >= out.size(1))
    return;
  const int32_t N = static_cast<int32_t>(sx[x]);
  const int32_t M = static_cast<int32_t>(sy[y]);
  if (N <= 0 || M <= 0)
    return;

  __shared__ scalar_t cost_buf[3][MAX_DIAG_LEN];
  __shared__ uint16_t len_buf[3][MAX_DIAG_LEN];
  int32_t alpha = 0; // Last diagonal
  int32_t beta = 1;  // Second to last diagonal
  int32_t gamma = 2; // Buffer for the last diagonal

  const auto distances_xy = distances[x][y];

  if (threadIdx.x == 0) {
    cost_buf[gamma][0] = distances_xy[0][0];
    len_buf[gamma][0] = 1;
  }
  __syncthreads();
  const int32_t temp = beta;
  beta = alpha;
  alpha = gamma;
  gamma = temp;

  const scalar_t max_val = std::numeric_limits<scalar_t>::max();
  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      const scalar_t dist = distances_xy[i][j];
      const scalar_t c_up = (i > 0) ? cost_buf[alpha][j] : max_val;
      const scalar_t c_left = (j > 0) ? cost_buf[alpha][j - 1] : max_val;
      const scalar_t c_diag = (i > 0 && j > 0) ? cost_buf[beta][j - 1] : max_val;
      if (step_pattern == 2) {
        const scalar_t two = static_cast<scalar_t>(2);
        const scalar_t v_diag = c_diag + two * dist;
        const scalar_t v_left = c_left + dist;
        const scalar_t v_up = c_up + dist;
        scalar_t min_cost;
        if (v_diag <= v_left && v_diag <= v_up) {
          min_cost = v_diag;
        } else if (v_left <= v_up) {
          min_cost = v_left;
        } else {
          min_cost = v_up;
        }
        cost_buf[gamma][j] = min_cost;
      } else {
        scalar_t min_cost;
        uint16_t parent_len;
        if (c_diag <= c_left && c_diag <= c_up) {
          min_cost = c_diag;
          parent_len = len_buf[beta][j - 1];
        } else if (c_left <= c_up) {
          min_cost = c_left;
          parent_len = len_buf[alpha][j - 1];
        } else {
          min_cost = c_up;
          parent_len = len_buf[alpha][j];
        }
        cost_buf[gamma][j] = dist + min_cost;
        len_buf[gamma][j] = static_cast<uint16_t>(parent_len + 1);
      }
    }
    __syncthreads();

    const int32_t temp = beta;
    beta = alpha;
    alpha = gamma;
    gamma = temp;
  }

  if (threadIdx.x == 0) {
    const scalar_t final_cost = cost_buf[alpha][M - 1];
    scalar_t result;
    if (step_pattern == 2) {
      result = final_cost / static_cast<scalar_t>(N + M);
    } else {
      const uint16_t path_len = len_buf[alpha][M - 1];
      result = final_cost / static_cast<scalar_t>(path_len);
    }
    out[x][y] = result;
    if (symmetric)
      out[y][x] = result;
  }
}

template <typename distances_t, typename sx_t>
void dtw_batch_cuda_impl(
    Tensor& out, const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric, int64_t step_pattern) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);
  const dim3 num_blocks = symmetric ? dim3(static_cast<unsigned int>(nx * (nx - 1) / 2)) : dim3(nx, ny);
  const int64_t max_diag = max_x < max_y ? max_x : max_y;
  const int num_threads = max_diag > 1024 ? 1024 : static_cast<int>(max_diag);
  const bool needs_64bit = nx * ny * max_x * max_y > std::numeric_limits<int32_t>::max();
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();

  if (needs_64bit) {
    dtw_kernel<distances_t, sx_t, int64_t><<<num_blocks, num_threads, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor64<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric,
        static_cast<int>(step_pattern));
  } else {
    dtw_kernel<distances_t, sx_t, int32_t><<<num_blocks, num_threads, 0, stream>>>(
        packed_accessor32<distances_t, 2>(out),
        packed_accessor32<distances_t, 4>(distances),
        packed_accessor32<sx_t, 1>(sx),
        packed_accessor32<sx_t, 1>(sy),
        symmetric,
        static_cast<int>(step_pattern));
  }
}

Tensor dtw_batch_cuda(
    const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric, int64_t step_pattern) {
  STD_TORCH_CHECK(distances.dim() == 4, "distances must be a 4D tensor");

  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(sx.dim() == 1 && sy.dim() == 1, "sx and sy must be 1D tensors");
  STD_TORCH_CHECK(sx.is_cuda() && sy.is_cuda(), "sx and sy must be on CUDA");
  STD_TORCH_CHECK(
      sx.size(0) == nx && sy.size(0) == ny, "sx and sy sizes must match the first two dimensions of distances");
  STD_TORCH_CHECK(!symmetric || nx == ny, "symmetric dtw_batch requires distances.size(0) == distances.size(1)");
  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(max_y <= MAX_DIAG_LEN, "Last dimension > 1638: too large to use CUDA shared memory");
  STD_TORCH_CHECK(
      step_pattern != 1 || max_x + max_y - 1 <= std::numeric_limits<uint16_t>::max(),
      "Sum of sequence lengths exceeds uint16_t path-length capacity");
  STD_TORCH_CHECK(sy.scalar_type() == sx.scalar_type(), "sx and sy dtypes do not match");

  Tensor out = torch::stable::new_zeros(distances, {nx, ny});
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
  Tensor sx = torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sx, distances.size(0));
  Tensor sy = torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sy, distances.size(1));
  Tensor result = dtw_batch_cuda(
      torch::stable::view(distances, {1, 1, distances.size(0), distances.size(1)}), sx, sy, false, step_pattern);
  return torch::stable::view(result, {});
}

// Wavefront-parallel computation of the full NxM cost matrix in global memory.
template <typename scalar_t>
__global__ void dtw_cost_matrix_kernel(
    PackedTensorAccessor32<scalar_t, 2> cost, const PackedTensorAccessor32<scalar_t, 2> distances,
    int step_pattern) {
  const int32_t N = cost.size(0);
  const int32_t M = cost.size(1);

  if (threadIdx.x == 0) {
    cost[0][0] = distances[0][0];
  }
  __syncthreads();

  for (int32_t diag = 1; diag <= N + M - 2; diag++) {
    const int32_t start_i = min(diag, N - 1);
    const int32_t start_j = max(0, diag - start_i);
    const int32_t length = start_i - max(0, diag - M + 1) + 1;

    for (int32_t k = threadIdx.x; k < length; k += blockDim.x) {
      const int32_t i = start_i - k;
      const int32_t j = start_j + k;
      const scalar_t dist = distances[i][j];
      if (i == 0) {
        cost[0][j] = cost[0][j - 1] + dist;
      } else if (j == 0) {
        cost[i][0] = cost[i - 1][0] + dist;
      } else if (step_pattern == 2) {
        const scalar_t two = static_cast<scalar_t>(2);
        const scalar_t v_diag = cost[i - 1][j - 1] + two * dist;
        const scalar_t v_left = cost[i][j - 1] + dist;
        const scalar_t v_up = cost[i - 1][j] + dist;
        scalar_t min_cost;
        if (v_diag <= v_left && v_diag <= v_up) {
          min_cost = v_diag;
        } else if (v_left <= v_up) {
          min_cost = v_left;
        } else {
          min_cost = v_up;
        }
        cost[i][j] = min_cost;
      } else {
        const scalar_t c_diag = cost[i - 1][j - 1];
        const scalar_t c_left = cost[i][j - 1];
        const scalar_t c_up = cost[i - 1][j];
        scalar_t min_cost;
        if (c_diag <= c_left && c_diag <= c_up) {
          min_cost = c_diag;
        } else if (c_left <= c_up) {
          min_cost = c_left;
        } else {
          min_cost = c_up;
        }
        cost[i][j] = dist + min_cost;
      }
    }
    __syncthreads();
  }
}

// Single-thread traceback from (N-1, M-1) to (0,0). Writes the path forward, reverses in-place,
// and outputs the path length and normalized cost.
template <typename scalar_t>
__global__ void dtw_traceback_kernel(
    int64_t* __restrict__ path, int64_t* __restrict__ path_length, scalar_t* __restrict__ cost_out,
    const PackedTensorAccessor32<scalar_t, 2> cost, const PackedTensorAccessor32<scalar_t, 2> distances,
    int step_pattern) {
  const int32_t N = cost.size(0);
  const int32_t M = cost.size(1);
  int32_t i = N - 1;
  int32_t j = M - 1;
  int32_t len = 0;

  path[0] = i;
  path[1] = j;
  len++;

  while (i > 0 && j > 0) {
    if (step_pattern == 2) {
      const scalar_t d = distances[i][j];
      const scalar_t c_diag = cost[i - 1][j - 1] + static_cast<scalar_t>(2) * d;
      const scalar_t c_left = cost[i][j - 1] + d;
      const scalar_t c_up = cost[i - 1][j] + d;
      if (c_diag <= c_left && c_diag <= c_up) {
        i--;
        j--;
      } else if (c_left <= c_up) {
        j--;
      } else {
        i--;
      }
    } else {
      const scalar_t c_up = cost[i - 1][j];
      const scalar_t c_left = cost[i][j - 1];
      const scalar_t c_diag = cost[i - 1][j - 1];
      if (c_diag <= c_left && c_diag <= c_up) {
        i--;
        j--;
      } else if (c_left <= c_up) {
        j--;
      } else {
        i--;
      }
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
  const scalar_t final_cost = cost[N - 1][M - 1];
  if (step_pattern == 2) {
    *cost_out = final_cost / static_cast<scalar_t>(N + M);
  } else {
    *cost_out = final_cost / static_cast<scalar_t>(len);
  }
}

template <typename scalar_t>
void dtw_cost_and_path_cuda_impl(
    Tensor& cost, Tensor& path, Tensor& path_length_t, Tensor& cost_out, const Tensor& distances,
    int64_t step_pattern) {
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
  const int64_t max_diag = N < M ? N : M;
  const int num_threads = max_diag > 1024 ? 1024 : static_cast<int>(max_diag);

  dtw_cost_matrix_kernel<scalar_t><<<1, num_threads, 0, stream>>>(
      packed_accessor32<scalar_t, 2>(cost),
      packed_accessor32<scalar_t, 2>(distances),
      static_cast<int>(step_pattern));

  dtw_traceback_kernel<scalar_t><<<1, 1, 0, stream>>>(
      reinterpret_cast<int64_t*>(path.data_ptr()),
      reinterpret_cast<int64_t*>(path_length_t.data_ptr()),
      reinterpret_cast<scalar_t*>(cost_out.data_ptr()),
      packed_accessor32<scalar_t, 2>(cost),
      packed_accessor32<scalar_t, 2>(distances),
      static_cast<int>(step_pattern));
}

std::tuple<Tensor, Tensor> dtw_cost_and_path_cuda(const Tensor& distances, int64_t step_pattern) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");

  Tensor cost = torch::stable::empty_like(distances);
  const int64_t max_path_len = N + M - 1;
  Tensor path = torch::stable::new_empty(distances, {max_path_len, 2}, std::make_optional(torch::headeronly::ScalarType::Long));
  Tensor path_length_t = torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  Tensor cost_out = torch::stable::new_empty(distances, {});

  THO_DISPATCH_V2(
      distances.scalar_type(),
      "dtw_cost_and_path_cuda",
      AT_WRAP([&] { dtw_cost_and_path_cuda_impl<scalar_t>(cost, path, path_length_t, cost_out, distances, step_pattern); }),
      AT_ALL_TYPES,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::BFloat16);

  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();
  int64_t path_len;
  cudaMemcpyAsync(&path_len, path_length_t.data_ptr(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return {cost_out, torch::stable::narrow(path, 0, 0, path_len)};
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
