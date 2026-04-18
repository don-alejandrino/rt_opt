import numpy as np


class SphereObjectiveFunction:
    def __init__(self) -> None:
        self.call_counter = 0

    def __call__(self, x: np.ndarray) -> float:
        self.call_counter += 1
        return np.square(x).sum()
