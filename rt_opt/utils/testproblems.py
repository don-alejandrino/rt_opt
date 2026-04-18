"""Module defining standard test problems for optimization algorithms.

For more details on the individual test problems, see
https://en.wikipedia.org/wiki/Test_functions_for_optimization.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Bounds:
    """Class to hold rectangular domain bounds.

    :param lower: Lower bound.
    :param upper: Upper bound.
    """

    lower: np.ndarray
    upper: np.ndarray


@dataclass
class SingleMinimum:
    """Class to hold known function minimum information if there is a single minimum.

    :param x: Location of the minimum.
    :param f: Function value at the minimum. If the true minimum value cannot be
           determined exactly, a range within which the true minimum must be located can
           be given as a tuple.
    """

    x: np.ndarray
    f: float | tuple[float, float]


@dataclass
class MultiMinimum:
    """Class to hold known function minimum information if there are multiple minima.

    :param x: Location of the minima.
    :param f: Function value at all the minima. If the true minimum value cannot be
           determined exactly, a range within which the true minimum must be located can
           be given as a tuple.
    """

    x: list[np.ndarray]
    f: float | tuple[float, float]


class TestProblem:
    """Base class for test problems."""

    __test__ = False  # Prevent pytest from trying to collect this class

    def __init__(self, ndims: int) -> None:
        """Initialize test problem.

        :param ndims: Number of dimensions.

        """
        self._ndims = ndims

    def __repr__(self) -> str:
        """Return string representation of the test problem."""
        return f"{self.__class__.__name__}({self.ndims})"

    def f(self, x: np.ndarray) -> float:
        """Objective function definition."""
        raise NotImplementedError

    @property
    def ndims(self) -> int:
        """Return the problem's number of dimensions."""
        return self._ndims

    @property
    def bounds(self) -> Bounds:
        """Return the problem's domain bounds."""
        raise NotImplementedError

    @property
    def min(self) -> SingleMinimum | MultiMinimum:
        """Return known x_min and f(x_min)."""
        raise NotImplementedError


class Rastrigin(TestProblem):
    """Rastrigin test problem."""

    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        return (
            10.0 * self.ndims + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x))
        ).astype(float)

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-5.12, self.ndims), upper=np.repeat(5.12, self.ndims)
        )

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.repeat(0, self.ndims), f=0)


class Ackley(TestProblem):
    """Ackley test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
            + np.exp(1)
            + 20
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-5, self.ndims), upper=np.repeat(5, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.repeat(0, self.ndims), f=0)


class Sphere(TestProblem):
    """Sphere test problem."""

    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        return np.sum(np.square(x))

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-3, self.ndims), upper=np.repeat(3, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.repeat(0, self.ndims), f=0)


class Rosenbrock(TestProblem):
    """Rosenbrock test problem."""

    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        x_ret = x[:-1]
        x_adv = np.roll(x, -1)[:-1]
        return (100 * np.square(x_adv - np.square(x_ret)) + np.square(1 - x_ret)).sum()

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-3, self.ndims), upper=np.repeat(3, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.repeat(1, self.ndims), f=0)


class Beale(TestProblem):
    """Beale test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * (x2**2)) ** 2
            + (2.625 - x1 + x1 * (x2**3)) ** 2
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-4.5, self.ndims), upper=np.repeat(4.5, self.ndims)
        )

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([3, 0.5]), f=0)


class GoldsteinPrice(TestProblem):
    """Goldstein-Price test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            1
            + (x1 + x2 + 1) ** 2
            * (19 - 14 * x1 + 3 * (x1**2) - 14 * x2 + 6 * x1 * x2 + 3 * (x2**2))
        ) * (
            30
            + (2 * x1 - 3 * x2) ** 2
            * (18 - 32 * x1 + 12 * (x1**2) + 48 * x2 - 36 * x1 * x2 + 27 * (x2**2))
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-2, self.ndims), upper=np.repeat(2, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([0, -1]), f=3)


class Booth(TestProblem):
    """Booth test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-10, self.ndims), upper=np.repeat(10, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([1, 3]), f=0)


class Bukin6(TestProblem):
    """Bukin N.6 test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return 100 * np.sqrt(np.abs(x2 - 0.01 * (x1**2))) + 0.01 * np.abs(x1 + 10)

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.array([-15, -3]), upper=np.array([-5, 3]))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([-10, 1]), f=0)


class Matyas(TestProblem):
    """Matyas test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-10, self.ndims), upper=np.repeat(10, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([0, 0]), f=0)


class Levi13(TestProblem):
    """Levi N.13 test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            np.sin(3 * np.pi * x1) ** 2
            + (x1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x2) ** 2)
            + (x2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x2) ** 2)
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-10, self.ndims), upper=np.repeat(10, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([1, 1]), f=0)


class Himmelblau(TestProblem):
    """Himmelblau test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-5, self.ndims), upper=np.repeat(5, self.ndims))

    @property
    def min(self) -> MultiMinimum:
        return MultiMinimum(
            x=[
                np.array([3, 2]),
                np.array([-2.805118086953, 3.131312518250]),
                np.array([-3.779310253378, -3.283185991286]),
                np.array([3.584428340330, -1.848126526964]),
            ],
            f=0,
        )


class ThreeHumpCamel(TestProblem):
    """Three-Hump Camel test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return 2 * (x1**2) - 1.05 * (x1**4) + 1 / 6 * (x1**6) + x1 * x2 + x2**2

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-5, self.ndims), upper=np.repeat(5, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([0, 0]), f=0)


class Easom(TestProblem):
    """Easom test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-100, self.ndims), upper=np.repeat(100, self.ndims)
        )

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([np.pi, np.pi]), f=-1)


class CrossInTray(TestProblem):
    """Cross-in-Tray test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            -0.0001
            * (
                np.abs(
                    np.sin(x1)
                    * np.sin(x2)
                    * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
                )
                + 1
            )
            ** 0.1
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-10, self.ndims), upper=np.repeat(10, self.ndims))

    @property
    def min(self) -> MultiMinimum:
        return MultiMinimum(
            x=[
                np.array([1.349406608602084, 1.349406608602084]),
                np.array([1.349406608602084, -1.349406608602084]),
                np.array([-1.349406608602084, 1.349406608602084]),
                np.array([-1.349406608602084, -1.349406608602084]),
            ],
            f=-2.062611870822739,
        )


class Eggholder(TestProblem):
    """Eggholder test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return -(x2 + 47) * np.sin(np.sqrt(abs(0.5 * x1 + (x2 + 47)))) - x1 * np.sin(
            np.sqrt(abs(x1 - (x2 + 47)))
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-512, self.ndims), upper=np.repeat(512, self.ndims)
        )

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([512, 404.231805123817]), f=-959.6406627208508)


class Hoelder(TestProblem):
    """Hoelder test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return -abs(
            np.sin(x1) * np.cos(x2) * np.exp(abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-10, self.ndims), upper=np.repeat(10, self.ndims))

    @property
    def min(self) -> MultiMinimum:
        return MultiMinimum(
            x=[
                np.array([8.05502347605272, 9.66459002316199]),
                np.array([-8.05502347605272, 9.66459002316199]),
                np.array([8.05502347605272, -9.66459002316199]),
                np.array([-8.05502347605272, -9.66459002316199]),
            ],
            f=-19.208502567886747,
        )


class McCormick(TestProblem):
    """McCormick test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.array([-1.5, -3]), upper=np.array([4, 4]))

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(
            x=np.array([-0.5471975602214493, -1.547197559268372]), f=-1.913222954981036
        )


class Schaffer2(TestProblem):
    """Schaffer N.2 test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            0.5
            + (np.sin(x1**2 - x2**2) ** 2 - 0.5) / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-100, self.ndims), upper=np.repeat(100, self.ndims)
        )

    @property
    def min(self) -> SingleMinimum:
        return SingleMinimum(x=np.array([0, 0]), f=0)


class Schaffer4(TestProblem):
    """Schaffer N.4 test problem."""

    def __init__(self) -> None:
        super().__init__(2)

    def f(self, x: np.ndarray) -> float:
        x1 = x[0]
        x2 = x[1]
        return (
            0.5
            + (np.cos(np.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5)
            / (1 + 0.001 * (x1**2 + x2**2)) ** 2
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            lower=np.repeat(-100, self.ndims), upper=np.repeat(100, self.ndims)
        )

    @property
    def min(self) -> MultiMinimum:
        return MultiMinimum(
            x=[
                np.array([0, 1.253131828792882]),
                np.array([0, -1.253131828792882]),
                np.array([1.253131828792882, 0]),
                np.array([-1.253131828792882, 0]),
            ],
            f=0.2925786320359805,
        )


class StyblinskiTang(TestProblem):
    """Styblinski-Tang test problem."""

    def __init__(self, ndims: int) -> None:
        super().__init__(ndims)

    def f(self, x: np.ndarray) -> float:
        return 0.5 * (np.power(x, 4) - 16 * np.square(x) + 5 * x).sum()

    @property
    def bounds(self) -> Bounds:
        return Bounds(lower=np.repeat(-5, self.ndims), upper=np.repeat(5, self.ndims))

    @property
    def min(self) -> SingleMinimum:
        # Depending on the choice of ndims, f(x_min) slightly varies. Hence, in general,
        # we can give only a range within which f(x_min) is located
        return SingleMinimum(
            x=np.repeat(-2.903534, self.ndims),
            f=(-39.16617 * self.ndims, -39.16616 * self.ndims),
        )
