"""Utility functions for input/output handling."""

import warnings

import numpy as np
from numpy._typing import ArrayLike


def prepare_bounds(
    bounds: ArrayLike | None, n_dims: int
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare bounds object.

    Check size and validity of a rectangular bounds object, and turn it into the
    required format for the following calculations.

    :param bounds: Rectangular bounds object.
    :param n_dims: Dimensionality of the problem. If not known a priori, set to None.
    :return: Lower and upper bounds as a tuple.

    """
    if bounds is not None:
        bounds = np.array(bounds)
        if bounds.shape != (n_dims, 2):
            err_msg = (
                "`bounds` has wrong shape. Expected shape is (n_dims, 2), where "
                "n_dims is the dimensionality of the problem. "
                f"Got shape {bounds.shape}."
            )
            raise ValueError(err_msg)
        bound_lower = bounds[:, 0]
        bound_upper = bounds[:, 1]
        if (bound_upper <= bound_lower).any():
            err_msg = "Upper bound must always be larger than lower bound."
            raise ValueError(err_msg)

        return bound_lower, bound_upper

    return np.repeat(-np.inf, n_dims), np.repeat(np.inf, n_dims)


def prepare_x0(
    x0: ArrayLike,
    n_bacteria_per_dim: int,
    max_dims: int,
    n_reduced_dims_eff: int,
) -> np.ndarray:
    """Prepare initial conditions object.

    Check and prepare initial conditions object x0. If x0 is a vector, that is, if it
    has the shape (n_dims,) it is duplicated times the total number of bacteria, which
    is given by
    i)  n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= `max_dims` or
    ii) n_bacteria = n_bacteria_per_dim ** n_reduced_dims_eff if n_dims > `max_dims`.

    :param x0: Initial conditions object. Must have the shape (n_bacteria, n_dims) or
           (n_dims,).
    :param n_bacteria_per_dim: Number of bacteria for each dimension.
    :param max_dims: Maximum dimension of problems to be solved without using Sequential
           Random Embeddings.
    :param n_reduced_dims_eff: Number of effective reduced dimensions used by the
           Sequential Random Embeddings algorithm
    :return: Initial conditions for all bacteria [array of shape (n_bacteria, n_dims)].

    """
    x0 = np.array(x0)
    if len(x0.shape) == 1:
        n_dims = x0.shape[0]
        n_bacteria = (
            n_bacteria_per_dim**n_dims
            if n_dims <= max_dims
            else n_bacteria_per_dim**n_reduced_dims_eff
        )
        x0_population = np.tile(x0, (n_bacteria, 1))
    elif len(x0.shape) == 2:  # noqa: PLR2004
        n_dims = x0.shape[1]
        n_bacteria = x0.shape[0]
        n_bacteria_target = (
            n_bacteria_per_dim**n_dims
            if n_dims <= max_dims
            else n_bacteria_per_dim**n_reduced_dims_eff
        )
        if n_bacteria != n_bacteria_target:
            warnings.warn(
                "The number of bacteria given by `x0` does not match the number of "
                "bacteria given by the relation "
                "``n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= max_dims "
                "else n_bacteria_per_dim ** (n_reduced_dims + 1)``. This relation "
                f"implies that n_bacteria = {n_bacteria_target}, whereas the choice of "
                f"`x0` implies that n_bacteria = {n_bacteria}. "
                f"Using n_bacteria = {n_bacteria}.",
                stacklevel=2,
            )
        x0_population = x0.copy()
    else:
        err_msg = (
            "`x0` must be an array of either the shape (n_bacteria, n_dims) or "
            "(n_dims,)."
        )
        raise ValueError(err_msg)

    return x0_population


def pad_trace(trace: np.ndarray, target_length: int) -> np.ndarray:
    """Pad single-bacteria trace to given target length by repeating the last entry.

    :param trace: Single-bacteria trace.
    :param target_length: Desired length of the trace after padding.
    :return: Padded trace with the last entry repeated as often as necessary to reach
             the target length.

    """
    current_length = trace.shape[0]
    padding_length = target_length - current_length

    return np.pad(trace, ((0, padding_length), (0, 0)), mode="edge")
