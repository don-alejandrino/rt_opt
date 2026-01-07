"""Protocol definitions for optimizers."""

from typing import Protocol

import numpy as np

from rt_opt.search.search_output import SingleSearchOutput
from rt_opt.utils.types import ObjectiveFunctionType, ProjectionCallbackType


class OptimizerProtocol(Protocol):
    """Protocol for optimizers."""

    def run(
        self,
        f: ObjectiveFunctionType,
        x0_population: np.ndarray,
        projection_callback: ProjectionCallbackType,
        projection_callback_population: ProjectionCallbackType,
    ) -> SingleSearchOutput:
        """Run the optimization routine.

        :param f: Objective function. Must accept its argument `x` as numpy array.
        :param x0_population: Initial condition. Must have the shape
               (n_bacteria, n_dims).
        :param projection_callback: Bounds projection callback, see description of
               parameter `projection_callback` in :func:`search.local_search.bfgs_b`.
        :param projection_callback_population: Same as `projection_callback`, but for
               the whole bacteria population. That is, in- and outputs must have
               the shape (n_bacteria, n_dims).
        :return: A `SingleSearchOutput` instance containing the optimization results.

        """
        ...
