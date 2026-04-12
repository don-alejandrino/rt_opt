import re
from collections.abc import Sequence

import numpy as np
import pytest

from rt_opt.utils.io import pad_trace, prepare_bounds, prepare_x0


@pytest.mark.parametrize(
    ("bounds", "ndims"),
    [
        ([[0, 1], [0, 2]], 2),
        (([0, 1], [0, 2]), 2),
        (((0, 1), (0, 2)), 2),
        (np.array([[0, 1], [0, 2]]), 2),
        ([[0, 1]], 2),  # invalid shape (bounds is 1D, expected 2D)
        ([[1, 0], [0, 2]], 2),  # lower bound greater than upper bound
        (None, 3),
    ],
    ids=[
        "list of lists as bounds",
        "tuple of lists as bounds",
        "tuple of tuples as bounds",
        "numpy array as bounds",
        "invalid shape of bounds",
        "invalid values of bounds",
        "None as bounds",
    ],
)
def test_prepare_bounds(bounds: Sequence[Sequence[int]], ndims: int) -> None:
    if bounds is not None and len(bounds) != ndims:
        with pytest.raises(ValueError, match=r"`bounds` has wrong shape."):
            prepare_bounds(bounds, ndims)
    elif bounds is not None and any(b[1] <= b[0] for b in bounds):
        with pytest.raises(
            ValueError, match=r"Upper bound must always be larger than lower bound."
        ):
            prepare_bounds(bounds, ndims)
    else:
        bound_lower, bound_upper = prepare_bounds(bounds, ndims)
        assert np.array_equal(
            bound_lower,
            np.array([b[0] for b in bounds])
            if bounds is not None
            else np.repeat(-np.inf, ndims),
        )
        assert np.array_equal(
            bound_upper,
            np.array([b[1] for b in bounds])
            if bounds is not None
            else np.repeat(np.inf, ndims),
        )


@pytest.mark.parametrize(
    "x0",
    [
        [0, 1],
        (0, 1),
        np.array([0, 1]),
        [[0, 1] for _ in range(9)],
        [[0, 1] for _ in range(6)],
        [[0, 1, 2, 3] for _ in range(27)],
        [[0, 1, 2, 3] for _ in range(81)],
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    ],
    ids=[
        "single x0 as list [2D]",
        "single x0 as tuple [2D]",
        "single x0 as numpy array [2D]",
        "multi-bacteria x0 [2D]",
        "multi-bacteria x0 with unexpected number of bacteria [2D]",
        "multi-bacteria x0 [4D]",
        "multi-bacteria x0 with unexpected number of bacteria [4D]",
        "invalid shape of x0",
    ],
)
def test_prepare_x0(x0: Sequence[int | Sequence[int]]) -> None:
    if isinstance(x0, np.ndarray) and len(x0.shape) > 2:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`x0` must be an array of either the shape (n_bacteria, n_dims) or "
                "(n_dims,)."
            ),
        ):
            prepare_x0(x0, n_bacteria_per_dim=3, max_dims=3, n_reduced_dims_eff=3)
        return

    if len(x0) in [6, 81]:
        with pytest.warns(
            UserWarning,
            match=r"The number of bacteria given by `x0` does not match the number of "
            "bacteria given by the relation",
        ):
            x0_population = prepare_x0(
                x0, n_bacteria_per_dim=3, max_dims=3, n_reduced_dims_eff=3
            )
    else:
        x0_population = prepare_x0(
            x0, n_bacteria_per_dim=3, max_dims=3, n_reduced_dims_eff=3
        )

    first_xo_element = x0[0]
    if isinstance(first_xo_element, Sequence):  # multi-bacteria x0
        assert x0_population.shape[0] == len(x0)
        assert x0_population.shape[1] == len(first_xo_element)
    else:  # single-bacteria x0
        assert x0_population.shape[0] == 3 ** len(x0)
        assert x0_population.shape[1] == len(x0)


def test_pad_trace() -> None:
    trace = np.array([[1, 2], [3, 4], [5, 6]])
    padded_trace = pad_trace(trace, target_length=5)
    expected_trace = np.array([[1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
    assert np.array_equal(padded_trace, expected_trace)
