"""Data classes for search outputs."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult


@dataclass
class SingleSearchOutput:
    """Output of a single-bacteria search.

    :param x_best: Best x found.
    :param f_best: Corresponding objective function value.
    :param nfev: Number of objective function evaluations taken.
    :param nit: Number of iterations during the search algorithm.
    :param success: Whether the search algorithm finished successfully (e.g., whether
           absolute and relative tolerances were met).
    :param trace: Trace containing all visited points of x during the search
    """

    x_best: np.ndarray
    f_best: float
    nfev: int
    nit: int
    success: bool
    trace: np.ndarray | None

    def to_optimize_result(self) -> OptimizeResult:
        """Convert to an OptimizeResult instance."""
        res_dict = {
            "success": self.success,
            "x": self.x_best,
            "fun": self.f_best,
            "nfev": self.nfev,
            "nit": self.nit,
        }
        if self.trace is not None:
            res_dict["trace"] = self.trace

        return OptimizeResult(**res_dict)


@dataclass
class MultiSearchOutput:
    """Output of a multi-bacteria search.

    :param x_best: Best x found for each bacterium.
    :param f_best: Corresponding objective function values.
    :param nfev: Number of objective function evaluations taken.
    :param nit: Number of iterations during the search algorithm.
    :param success: Whether the search algorithm finished successfully (e.g., whether
           absolute and relative tolerances were met).
    :param trace: Trace containing all visited points of x during the search
    """

    x_best: np.ndarray
    f_best: np.ndarray
    nfev: int
    nit: int
    success: bool
    trace: np.ndarray | None

    def to_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, bool, np.ndarray | None]:
        """Convert to tuple."""
        return (
            self.x_best,
            self.f_best,
            self.nfev,
            self.nit,
            self.success,
            self.trace,
        )


@dataclass
class LineSearchOutput:
    """Output of a line search.

    :param x: New position after the line search step.
    :param f: Objective function value at the new position.
    :param a: (Sub)optimal stepsize found by the algorithm.
    :param nfev: Number of objective function evaluations taken.
    :param nit: Number of iterations during the line search algorithm.
    :param success: Whether the line search exited successfully, i.e., whether a
           stepsize fulfilling the Armijo conditions was found.
    """

    x: np.ndarray
    f: float
    a: float
    nfev: int
    nit: int
    success: bool
