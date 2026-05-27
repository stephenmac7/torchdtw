"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import sys

import torch

from . import _C  # noqa: F401 # ty: ignore[unresolved-import]

__all__ = ["dtw", "dtw_cost_and_path", "dtw_batch"]

_STEP_PATTERNS = {"symmetric1": 1, "symmetric2": 2}


class CUDAOnWindowsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("torchdtw was built without CUDA support on Windows. Move your tensors to CPU first.")


def _check_no_cuda_on_windows(tensor: torch.Tensor) -> None:
    if sys.platform == "win32" and tensor.is_cuda:
        raise CUDAOnWindowsError


def dtw(distances: torch.Tensor, *, step_pattern: str = "symmetric1") -> torch.Tensor:
    """Compute the DTW cost of the given ``distances`` 2D tensor.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :param step_pattern: Step pattern to use: ``"symmetric1"`` or ``"symmetric2"``.
    :returns: A scalar tensor with the cost.
    """
    _check_no_cuda_on_windows(distances)
    return torch.ops.torchdtw.dtw.default(distances, _STEP_PATTERNS[step_pattern])


def dtw_cost_and_path(distances: torch.Tensor, *, step_pattern: str = "symmetric1") -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DTW cost and path in a single pass.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :param step_pattern: Step pattern to use: ``"symmetric1"`` or ``"symmetric2"``.
    :returns: A tuple ``(cost, path)`` where *cost* is a scalar tensor and *path* has shape ``(*, 2)``.
    """
    _check_no_cuda_on_windows(distances)
    return torch.ops.torchdtw.dtw_cost_and_path.default(distances, _STEP_PATTERNS[step_pattern])


def dtw_path(distances: torch.Tensor, *, step_pattern: str = "symmetric1") -> torch.Tensor:
    """Compute the DTW path of the given ``distances`` 2D tensor.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    :param step_pattern: Step pattern to use: ``"symmetric1"`` or ``"symmetric2"``.
    :returns: A 2D tensor of shape (*, 2) with the path indices.
    """
    _check_no_cuda_on_windows(distances)
    return torch.ops.torchdtw.dtw_path.default(distances, _STEP_PATTERNS[step_pattern])


def dtw_batch(
    distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool,
    step_pattern: str = "symmetric1",
) -> torch.Tensor:
    """Compute the batched DTW cost on the ``distances`` 4D tensor.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
        batches of sequences.
    :param sx: A 1D tensor of shape (n1,) representing the lengths of the sequences in the first batch.
    :param sy: A 1D tensor of shape (n2,) representing the lengths of the sequences in the second batch.
    :param symmetric: Whether or not the DTW is symmetric (i.e., the two batches are the same).
    :param step_pattern: Step pattern to use: ``"symmetric1"`` or ``"symmetric2"``.
    :returns: A 2D tensor of shape (n1, n2) with the costs.
    """
    _check_no_cuda_on_windows(distances)
    return torch.ops.torchdtw.dtw_batch.default(distances, sx, sy, symmetric, _STEP_PATTERNS[step_pattern])


@torch.library.register_fake("torchdtw::dtw")
def _(distances: torch.Tensor, step_pattern: int) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2)
    return torch.empty((), dtype=distances.dtype, layout=distances.layout, device=distances.device)


@torch.library.register_fake("torchdtw::dtw_batch")
def _(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, symmetric: bool, step_pattern: int) -> torch.Tensor:  # noqa: FBT001
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4)
    torch._check(sx.ndim == 1)
    torch._check(sy.ndim == 1)
    torch._check(not (sx.dtype.is_complex or sx.dtype.is_floating_point))
    torch._check(sx.dtype == sy.dtype)
    torch._check(isinstance(symmetric, bool))
    nx, ny, _, _ = distances.shape
    return torch.empty((nx, ny), dtype=distances.dtype, layout=distances.layout, device=distances.device)
