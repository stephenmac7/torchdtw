"""Compare torchdtw symmetric2 with dtw-python."""

import torch
from dtw import dtw as dtw_python
from hypothesis import given
from hypothesis import strategies as st

import torchdtw

DIM = st.integers(1, 50)


@given(x=DIM, y=DIM)
def test_symmetric2_distance(x: int, y: int) -> None:
    """Verify that torchdtw symmetric2 matches dtw-python's normalizedDistance."""
    d = torch.testing.make_tensor((x, y), dtype=torch.float64, device="cpu", low=0.0, high=10.0)

    result = torchdtw.dtw(d, step_pattern="symmetric2")
    alignment = dtw_python(d.numpy(), step_pattern="symmetric2")
    expected = alignment.distance / (x + y)

    torch.testing.assert_close(
        result,
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@given(x=DIM, y=DIM)
def test_symmetric2_path_cost(x: int, y: int) -> None:
    """Verify that the symmetric2 path cost is consistent with the returned distance."""
    d = torch.testing.make_tensor((x, y), dtype=torch.float64, device="cpu", low=0.1, high=10.0)

    path = torchdtw.dtw_path(d, step_pattern="symmetric2")
    distance = torchdtw.dtw(d, step_pattern="symmetric2")

    total_cost = torch.tensor(0.0, dtype=torch.float64)
    for k in range(len(path)):
        i, j = int(path[k, 0]), int(path[k, 1])
        if k == 0:
            weight = 1
        else:
            prev_i, prev_j = int(path[k - 1, 0]), int(path[k - 1, 1])
            weight = 2 if (i == prev_i + 1 and j == prev_j + 1) else 1
        total_cost += weight * d[i, j]

    torch.testing.assert_close(distance, total_cost / (x + y), rtol=1e-12, atol=1e-12)
