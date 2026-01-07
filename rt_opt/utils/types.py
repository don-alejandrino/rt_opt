"""Provide custom types."""

from collections.abc import Callable

import numpy as np

ProjectionCallbackType = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
ObjectiveFunctionType = Callable[[np.ndarray], float]
