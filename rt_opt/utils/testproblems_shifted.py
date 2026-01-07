"""Module defining shifted test problems for optimization algorithms.

Same test problems as in testproblems.py, except here all global minima that lie at
the origin are shifted by 1 along each axis to break spatial symmetry around the
origin.
"""

import numpy as np

import rt_opt.utils.testproblems as tp


class Rastrigin(tp.Rastrigin):
    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Ackley(tp.Ackley):
    def __init__(self) -> None:
        super().__init__()

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Sphere(tp.Sphere):
    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Rosenbrock(tp.Rosenbrock):
    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)


class Beale(tp.Beale):
    def __init__(self) -> None:
        super().__init__()


class GoldsteinPrice(tp.GoldsteinPrice):
    def __init__(self) -> None:
        super().__init__()


class Booth(tp.Booth):
    def __init__(self) -> None:
        super().__init__()


class Bukin6(tp.Bukin6):
    def __init__(self) -> None:
        super().__init__()


class Matyas(tp.Matyas):
    def __init__(self) -> None:
        super().__init__()

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Levi13(tp.Levi13):
    def __init__(self) -> None:
        super().__init__()


class Himmelblau(tp.Himmelblau):
    def __init__(self) -> None:
        super().__init__()


class ThreeHumpCamel(tp.ThreeHumpCamel):
    def __init__(self) -> None:
        super().__init__()

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Easom(tp.Easom):
    def __init__(self) -> None:
        super().__init__()


class CrossInTray(tp.CrossInTray):
    def __init__(self) -> None:
        super().__init__()


class Eggholder(tp.Eggholder):
    def __init__(self) -> None:
        super().__init__()


class Hoelder(tp.Hoelder):
    def __init__(self) -> None:
        super().__init__()


class McCormick(tp.McCormick):
    def __init__(self) -> None:
        super().__init__()


class Schaffer2(tp.Schaffer2):
    def __init__(self) -> None:
        super().__init__()

    def f(self, x: np.ndarray) -> float:
        return super().f(x - 1)

    @property
    def min(self) -> tp.SingleMinimum:
        m = super().min
        m.x = m.x + 1
        return m


class Schaffer4(tp.Schaffer4):
    def __init__(self) -> None:
        super().__init__()


class StyblinskiTang(tp.StyblinskiTang):
    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)
