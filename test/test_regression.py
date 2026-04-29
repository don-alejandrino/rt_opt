import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import OptimizeResult

import rt_opt.utils.testproblems_shifted as tps
from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.optimizer import OptimizationConfig, SequentialRandomEmbeddingsConfig
from rt_opt.optimizer import optimize
from rt_opt.utils.testproblems import TestProblem

N_RUNS = 100
TESTPROBLEMS_2D = [
    tps.Rastrigin(2),
    tps.Ackley(),
    tps.Sphere(2),
    tps.Rosenbrock(2),
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
    tps.Schaffer4(),
    tps.StyblinskiTang(2),
]
TESTPROBLEMS_15D = [
    tps.Rastrigin(15),
    tps.Sphere(15),
    tps.Rosenbrock(15),
    tps.StyblinskiTang(15),
]
ALL_TESTPROBLEMS = TESTPROBLEMS_2D + TESTPROBLEMS_15D
MINIMUM_FINDING_THRESHOLD = 1e-9

with Path("test/regression_baselines.yaml").open() as fh:
    BASELINE_METRICS = yaml.safe_load(fh)


def extract_bounds_from_testproblem(
    problem: TestProblem,
) -> tuple[np.ndarray, np.ndarray]:
    if problem.bounds.lower is not None:
        bounds_lower = problem.bounds.lower
    else:
        bounds_lower = np.repeat(-5, problem.ndims)
    if problem.bounds.upper is not None:
        bounds_upper = problem.bounds.upper
    else:
        bounds_upper = np.repeat(5, problem.ndims)
    return bounds_lower, bounds_upper


def run_test(problem: TestProblem, run_number: int) -> OptimizeResult:
    bounds = np.vstack(extract_bounds_from_testproblem(problem)).T
    # Initialize all seeds with the run number for reproducibility.
    config = OptimizationConfig(
        global_search=RunAndTumbleConfig(seed=run_number),
        embedding=SequentialRandomEmbeddingsConfig(seed=run_number),
        seed=run_number,
    )

    return optimize(problem.f, bounds=bounds, config=config)


def calculate_metrics(
    results: dict[str, np.ndarray], problem: TestProblem
) -> dict[str, float]:
    mean_nfev = results["nfev"].mean()
    std_nfev = results["nfev"].std()

    true_min_val = problem.min.f
    if isinstance(true_min_val, tuple):  # Only range for minimum value known
        errors = np.empty(len(results["f"]))
        for j, val in enumerate(results["f"]):
            if val < true_min_val[0]:
                errors[j] = val - true_min_val[0]
            elif val > true_min_val[1]:
                errors[j] = val - true_min_val[1]
            else:
                errors[j] = 0
    else:
        errors = results["f"] - true_min_val
    mae_f = np.abs(errors).mean()
    stdae_f = np.abs(errors).std()

    min_found = np.where(np.abs(errors) < MINIMUM_FINDING_THRESHOLD, 1, 0)
    finding_rate = min_found.sum() / len(min_found)

    return {
        "mean_nfev": mean_nfev,
        "std_nfev": std_nfev,
        "mae_f": mae_f,
        "stdae_f": stdae_f,
        "finding_rate": finding_rate,
    }


@pytest.mark.regression
@pytest.mark.parametrize(
    "problem",
    ALL_TESTPROBLEMS,
    ids=map(str, ALL_TESTPROBLEMS),
)
def test_regression(problem: TestProblem) -> None:
    with multiprocessing.Pool() as p:
        mp_output = p.map(partial(run_test, problem), range(N_RUNS))

    results = {
        "nfev": np.empty((N_RUNS,), dtype=int),
        "f": np.empty((N_RUNS,), dtype=float),
    }
    for i, result in enumerate(mp_output):
        results["nfev"][i] = result.nfev
        results["f"][i] = result.fun
    metrics = calculate_metrics(results, problem)
    baseline_metrics = BASELINE_METRICS[str(problem)]

    mean_nfev = metrics["mean_nfev"]
    mean_nfev_baseline = baseline_metrics["mean_nfev"]
    std_nfev_baseline = baseline_metrics["std_nfev"]
    assert mean_nfev < mean_nfev_baseline + 5 * std_nfev_baseline, (
        f"Metric 'mean_nfev' for problem {problem} is {mean_nfev:g}, which exceeds the "
        f"baseline of {mean_nfev_baseline:g} ± {std_nfev_baseline:g} by more than 5 "
        f"sigma."
    )

    mae_f = metrics["mae_f"]
    mae_f_baseline = baseline_metrics["mae_f"]
    stdae_f_baseline = baseline_metrics["stdae_f"]
    if mae_f_baseline > 0:
        assert mae_f < mae_f_baseline + 5 * stdae_f_baseline, (
            f"Metric 'mae_f' for problem {problem} is {mae_f:g}, which exceeds the "
            f"baseline of {mae_f_baseline:g} ± {stdae_f_baseline:g} by more than 5 "
            f"sigma."
        )

    finding_rate = metrics["finding_rate"]
    finding_rate_baseline = baseline_metrics["finding_rate"]
    if finding_rate_baseline > 0:
        assert finding_rate > finding_rate_baseline * 0.8, (
            f"Metric 'finding_rate' for problem {problem} is {finding_rate}, which is "
            f"less than 80% of the baseline of {finding_rate_baseline}."
        )
