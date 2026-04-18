import numpy as np
import pytest

from rt_opt.utils import testproblems as tp
from rt_opt.utils import testproblems_shifted as tps
from rt_opt.utils.testproblems import MultiMinimum, SingleMinimum, TestProblem


def run_test(problem: TestProblem) -> None:
    if isinstance(problem.min, MultiMinimum):
        if isinstance(problem.min.f, tuple):
            for val in problem.min.x:
                assert problem.min.f[0] < problem.f(val) < problem.min.f[1]
        else:
            for val in problem.min.x:
                assert problem.f(val) == pytest.approx(
                    problem.min.f, abs=np.finfo(float).eps
                )
    elif isinstance(problem.min, SingleMinimum):
        if isinstance(problem.min.f, tuple):
            assert problem.min.f[0] < problem.f(problem.min.x) < problem.min.f[1]
        else:
            assert problem.f(problem.min.x) == pytest.approx(
                problem.min.f, abs=np.finfo(float).eps
            )
    else:
        err_msg = f"`problem.min` has unexpected type {type(problem.min)}."
        raise TypeError(err_msg)


@pytest.mark.parametrize(
    "problem",
    [
        tp.Ackley(),
        tp.Beale(),
        tp.GoldsteinPrice(),
        tp.Booth(),
        tp.Bukin6(),
        tp.Matyas(),
        tp.Levi13(),
        tp.Himmelblau(),
        tp.ThreeHumpCamel(),
        tp.Easom(),
        tp.CrossInTray(),
        tp.Eggholder(),
        tp.Hoelder(),
        tp.McCormick(),
        tp.Schaffer2(),
        tp.Schaffer4(),
        tps.Ackley(),
        tps.Beale(),
        tps.GoldsteinPrice(),
        tps.Booth(),
        tps.Bukin6(),
        tps.Matyas(),
        tps.Levi13(),
        tps.Himmelblau(),
        tps.ThreeHumpCamel(),
        tps.Easom(),
        tps.CrossInTray(),
        tps.Eggholder(),
        tps.Hoelder(),
        tps.McCormick(),
        tps.Schaffer2(),
        tps.Schaffer4(),
    ],
)
def test_testproblem_2d(problem: TestProblem) -> None:
    run_test(problem)


@pytest.mark.parametrize("ndims", [2, 10, 100, 1000])
@pytest.mark.parametrize(
    "problem",
    [
        tp.Rastrigin,
        tp.Sphere,
        tp.Rosenbrock,
        tp.StyblinskiTang,
        tps.Rastrigin,
        tps.Sphere,
        tps.Rosenbrock,
        tps.StyblinskiTang,
    ],
)
def test_testproblem_nd(problem: type[TestProblem], ndims: int) -> None:
    run_test(problem(ndims))
