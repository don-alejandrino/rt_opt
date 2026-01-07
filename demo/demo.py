"""Demo script to compare global optimizers on benchmark test problems."""

import itertools
import logging
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import dlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy._typing import ArrayLike
from pandas import DataFrame
from scipy.optimize import OptimizeResult, differential_evolution, dual_annealing
from tqdm import tqdm

import rt_opt.utils.testproblems_shifted as tps
from rt_opt.optimizer import optimize
from rt_opt.utils.testproblems import MultiMinimum, SingleMinimum, TestProblem
from rt_opt.utils.types import ObjectiveFunctionType

SAVE_DIR = "demo/results"

logger = logging.getLogger(__name__)


class Metrics(Enum):
    """Enumeration of optimizer performance metrics."""

    RUNTIME_MEAN = "Running time (mean) [s]"
    RUNTIME_STD = "Running time (std) [s]"
    NFEV_MEAN = "No. objective function evaluations (mean)"
    NFEV_STD = "No. objective function evaluations (std)"
    RMSE_X = "RMSE in minimum position"
    AE_F_MEAN = "Absolute error in minimum value (mean)"
    AE_F_STD = "Absolute error in minimum value (std)"
    SUCCESS = "Finding rate of global minimum"


class DisplayMetrics(Enum):
    """Enumeration of optimizer performance metrics for display."""

    RUNTIME = "Running time (s)"
    NFEV = "No. objective function evaluations"
    RMSE_X = "RMSE in minimum position"
    MAE_F = "Absolute error in minimum value"
    SUCCESS = "Finding rate of global minimum"


def gridmap2d(
    fun: ObjectiveFunctionType,
    x_specs: tuple[float, float, int],
    y_specs: tuple[float, float, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D grid map of function values.

    :param fun: Objective function to be evaluated.
    :param x_specs: Specifications for x-axis grid, given as tuple
           (x_min, x_max, n_points).
    :param y_specs: Specifications for y-axis grid, given as tuple
           (y_min, y_max, n_points).
    :return: Grid arrays (arr_x, arr_y, arr_z) with arr_z being the function values
             evaluated on the grid defined by arr_x and arr_y.

    """
    grid_x = np.linspace(*x_specs)
    grid_y = np.linspace(*y_specs)
    arr_z = np.empty(len(grid_x) * len(grid_y))
    i = 0
    for y in grid_y:
        for x in grid_x:
            arr_z[i] = fun(np.array([x, y]))
            i += 1
    arr_x, arr_y = np.meshgrid(grid_x, grid_y)
    arr_z.shape = arr_x.shape
    return arr_x, arr_y, arr_z


def extract_bounds(problem: TestProblem) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper bounds from a test problem.

    :param problem: Test problem instance.
    :return: Tuple of (bounds_lower, bounds_upper) as numpy arrays.

    """
    if problem.bounds.lower is not None:
        bounds_lower = problem.bounds.lower
    else:
        bounds_lower = np.repeat(-5, problem.ndims)
    if problem.bounds.upper is not None:
        bounds_upper = problem.bounds.upper
    else:
        bounds_upper = np.repeat(5, problem.ndims)
    return bounds_lower, bounds_upper


def plot_bacteria_traces(  # noqa: PLR0913
    fig: Figure,
    axs: Any,  # noqa: ANN401
    problem: TestProblem,
    optimization_result: OptimizeResult,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    n_cols: int,
    n: int,
) -> None:
    """Plot bacteria traces for 2D problems.

    :param fig: Matplotlib figure object.
    :param axs: Matplotlib axes object.
    :param problem: Test problem instance.
    :param optimization_result: Optimization result from run-and-tumble optimizer.
    :param bounds_lower: Lower bounds of the problem.
    :param bounds_upper: Upper bounds of the problem.
    :param n_cols: Number of columns in the subplot grid.
    :param n: Index of the current subplot.

    """
    if problem.ndims != 2:  # noqa: PLR2004
        err_msg = (
            "Currently, plotting bacteria traces is supported for 2D problems only."
        )
        raise NotImplementedError(err_msg)

    plot_row = n // n_cols
    plot_col = n % n_cols
    x_arr, y_arr, z_arr = gridmap2d(
        problem.f,
        (bounds_lower[0], bounds_upper[0], 100),
        (bounds_lower[1], bounds_upper[1], 100),
    )
    cp = axs[plot_row, plot_col].contourf(x_arr, y_arr, z_arr, levels=20)
    axs[plot_row, plot_col].set_xlim([bounds_lower[0], bounds_upper[0]])
    axs[plot_row, plot_col].set_ylim([bounds_lower[1], bounds_upper[1]])
    fig.colorbar(cp, ax=axs[plot_row, plot_col])
    bacteria_traces: np.ndarray = optimization_result.trace
    for single_trace in bacteria_traces.transpose(1, 0, 2):
        axs[plot_row, plot_col].plot(
            single_trace[:, 0],
            single_trace[:, 1],
            "o",
            c="white",
            ms=0.4,
        )
    axs[plot_row, plot_col].set_xlabel("x")
    axs[plot_row, plot_col].set_ylabel("y")
    axs[plot_row, plot_col].set_title(
        f"Problem: {problem.__class__.__name__}", fontsize=10
    )


def finalize_bacteria_traces_plot(fig: Figure) -> None:
    """Finalize and save the bacteria traces plot.

    :param fig: Matplotlib figure object containing the plots of the traces.

    """
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.suptitle("Run-and-tumble bacteria traces", fontsize=14)
    fig_manager = plt.get_current_fig_manager()
    if fig_manager is not None:
        fig_manager.full_screen_toggle()
    fig.set_size_inches(25.6, 14.4)
    plt.savefig(Path(SAVE_DIR) / "bacteria_traces.png", bbox_inches="tight", dpi=150)
    plt.show()


def _update_optimizer_results(  # noqa: PLR0913
    optimizer_results: dict[str, dict[str, dict[str, list]]],
    runtime: float,
    nfev: int,
    x: np.ndarray,
    f: float | np.float64,
    problem_name: str,
    optimizer_name: str,
) -> None:
    optimizer_results[optimizer_name][problem_name]["runtime"].append(runtime)
    optimizer_results[optimizer_name][problem_name]["nfev"].append(nfev)
    optimizer_results[optimizer_name][problem_name]["x"].append(x)
    optimizer_results[optimizer_name][problem_name]["f"].append(f)


def calculate_optimizer_metrics(
    problems: list[TestProblem], n_runs: int, plot_traces: bool = False
) -> dict[str, dict[str, dict[str, list]]]:
    """Calculate optimizer performance metrics.

    Let the global optimizers rt_opt, scipy's differential_evolution, scipy's
    dual_annealing, and dlib's LIPO minimize a bunch of test functions and collect
    performance metrics.

    :param problems: List of test problems to be used.
    :param n_runs: How often each problem is solved by the different optimizers.
    :param plot_traces: Whether to plot bacteria traces for the rt_opt optimizer.
    :return: Performance metrics as a nested dictionary. The outer dictionary has
             optimizer names as keys. The second-level dictionaries have problem names
             as keys. The innermost dictionaries have metric names as keys and lists of
             metric values (one entry per run) as values.

    """
    optimizer_results: dict[str, dict[str, dict[str, list]]] = {
        "Run-and-Tumble": defaultdict(lambda: defaultdict(list)),
        "Differential Evolution": defaultdict(lambda: defaultdict(list)),
        "Dual Annealing": defaultdict(lambda: defaultdict(list)),
        "LIPO": defaultdict(lambda: defaultdict(list)),
    }

    n_plots = len(problems)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols)

    n_total_steps = len(problems) * len(optimizer_results) * n_runs

    logger.info("Collecting optimizer statistics.")
    with tqdm(total=n_total_steps) as pbar:
        for n, problem in enumerate(problems):
            name = problem.__class__.__name__
            bounds_lower, bounds_upper = extract_bounds(problem)

            # Run-and-tumble algorithm
            bounds = np.vstack((bounds_lower, bounds_upper)).T
            for m in range(n_runs):
                start = time.time()
                ret = optimize(problem.f, bounds=bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                _update_optimizer_results(
                    optimizer_results=optimizer_results,
                    runtime=runtime,
                    nfev=ret.nfev,
                    x=ret.x,
                    f=ret.fun,
                    problem_name=name,
                    optimizer_name="Run-and-Tumble",
                )

                if plot_traces and m == 0:  # Plot bacteria traces only once
                    plot_bacteria_traces(
                        fig=fig,
                        axs=axs,
                        problem=problem,
                        optimization_result=ret,
                        bounds_lower=bounds_lower,
                        bounds_upper=bounds_upper,
                        n_cols=n_cols,
                        n=n,
                    )

            # Differential Evolution algorithm
            bounds = tuple(np.vstack((bounds_lower, bounds_upper)).T)
            for _ in range(n_runs):
                start = time.time()
                ret = differential_evolution(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                _update_optimizer_results(
                    optimizer_results=optimizer_results,
                    runtime=runtime,
                    nfev=ret.nfev,
                    x=ret.x,
                    f=ret.fun,
                    problem_name=name,
                    optimizer_name="Differential Evolution",
                )

            # Dual Annealing algorithm
            bounds = np.vstack((bounds_lower, bounds_upper)).T
            for _ in range(n_runs):
                start = time.time()
                ret = dual_annealing(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                _update_optimizer_results(
                    optimizer_results=optimizer_results,
                    runtime=runtime,
                    nfev=ret.nfev,
                    x=ret.x,
                    f=ret.fun,
                    problem_name=name,
                    optimizer_name="Dual Annealing",
                )

            # LIPO algorithm
            nfev = 500 * problem.ndims

            def objective_function(*args: ArrayLike) -> float:
                """Objective function wrapper to match dlib's expected signature."""
                return problem.f(np.array(args))  # noqa: B023

            for _ in range(n_runs):
                start = time.time()
                ret_lp = dlib.find_min_global(  # type: ignore[reportAttributeAccessIssue]
                    objective_function,
                    bounds_lower.tolist(),
                    bounds_upper.tolist(),
                    nfev,
                )
                end = time.time()
                runtime = end - start
                pbar.update(1)
                _update_optimizer_results(
                    optimizer_results=optimizer_results,
                    runtime=runtime,
                    nfev=nfev,
                    x=ret_lp[0],
                    f=ret_lp[1],
                    problem_name=name,
                    optimizer_name="LIPO",
                )

    if plot_traces:
        finalize_bacteria_traces_plot(fig)

    return optimizer_results


def collect_statistics(
    problems: list[TestProblem],
    optimizer_results: dict[str, dict[str, dict[str, list]]],
    ndims: int,
    minimum_finding_threshold: float = 1e-9,
) -> pd.DataFrame:
    """Calculate and display optimizer performance statistics.

    :param problems: List of test problems to be used.
    :param optimizer_results: Optimizer performance metrics. See return value of
           `calculate_optimizer_metrics` for details.
    :param ndims: Dimension of the test problems.
    :param minimum_finding_threshold: Error threshold for considering the global minimum
           as found. That is, if |f_found - f_true_min| < minimum_finding_threshold, the
           minimum is considered found.

    """
    algo_names = list(optimizer_results.keys())
    problems_dict = {prob.__class__.__name__: prob for prob in problems}
    problem_names = list(problems_dict.keys())

    statistics_data = pd.DataFrame(columns=["Problem", "Metric", *algo_names])
    statistics_data["Problem"] = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, len(Metrics)) for x in problem_names
        )
    )
    statistics_data["Metric"] = [e.value for e in Metrics] * len(problem_names)

    table_data = pd.DataFrame(columns=["Problem", "Metric", *algo_names])
    table_data["Problem"] = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, len(DisplayMetrics)) for x in problem_names
        )
    )
    table_data["Metric"] = [e.value for e in DisplayMetrics] * len(problems)

    for algo, results in optimizer_results.items():
        for n, (problem_name, metrics) in enumerate(results.items()):
            mean_runtime = np.array(metrics["runtime"]).mean()
            std_runtime = np.array(metrics["runtime"]).std()
            mean_nfev = np.array(metrics["nfev"]).mean()
            std_nfev = np.array(metrics["nfev"]).std()

            true_min = problems_dict[problem_name].min
            if isinstance(true_min, MultiMinimum):  # More than one global minima
                distances = np.min(
                    np.square(
                        np.array(metrics["x"])[:, None, :]
                        - np.array(true_min.x)[None, :, :]
                    ).sum(axis=2),
                    axis=1,
                )
                rmse_x = np.sqrt(distances.mean())
            elif isinstance(true_min, SingleMinimum):
                rmse_x = np.sqrt(
                    np.square(np.array(metrics["x"]) - true_min.x).sum(axis=1).mean()
                )
            else:
                err_msg = f"`problem.min` has unexpected type {type(true_min)}."
                raise TypeError(err_msg)

            true_min_val = true_min.f
            if isinstance(true_min_val, tuple):  # Only range for minimum value known
                errors = np.empty(len(metrics["f"]))
                for j, val in enumerate(metrics["f"]):
                    if val < true_min_val[0]:
                        errors[j] = val - true_min_val[0]
                    elif val > true_min_val[1]:
                        errors[j] = val - true_min_val[1]
                    else:
                        errors[j] = 0
                mae_f = np.abs(errors).mean()
                stdae_f = np.abs(errors).std()
            else:
                errors = np.array(metrics["f"]) - true_min_val
                mae_f = np.abs(errors).mean()
                stdae_f = np.abs(errors).std()

            min_found = np.where(np.abs(errors) < minimum_finding_threshold, 1, 0)
            finding_rate = min_found.sum() / len(min_found)

            for idx, metric_val in enumerate(
                [
                    mean_runtime,
                    std_runtime,
                    mean_nfev,
                    std_nfev,
                    rmse_x,
                    mae_f,
                    stdae_f,
                    finding_rate,
                ]
            ):
                statistics_data.loc[len(Metrics) * n + idx, algo] = metric_val

            table_data.loc[len(DisplayMetrics) * n, algo] = (
                f"{mean_runtime:g} ± {std_runtime:g}"
            )
            table_data.loc[len(DisplayMetrics) * n + 1, algo] = (
                f"{mean_nfev:g} ± {std_nfev:g}"
            )
            table_data.loc[len(DisplayMetrics) * n + 2, algo] = f"{rmse_x:g}"
            table_data.loc[len(DisplayMetrics) * n + 3, algo] = (
                f"{mae_f:g} ± {stdae_f:g}"
            )
            table_data.loc[len(DisplayMetrics) * n + 4, algo] = f"{finding_rate:.3f}"

    # Export table data
    html_table = (
        table_data.sort_values(by=["Metric", "Problem"])
        .reset_index(drop=True)
        .to_html()
    )
    with (Path(SAVE_DIR) / f"optimizer_statistics_{ndims}D.html").open(
        "w", encoding="utf-8"
    ) as file:
        file.writelines('<meta charset="UTF-8">\n')
        file.write(html_table)

    return statistics_data


def plot_statistics(statistics_data: DataFrame, ndims: int) -> None:
    """Plot optimizer performance statistics.

    :param statistics_data: DataFrame containing the performance statistics.
    :param ndims: Dimension of the test problems.

    """
    fig, axs = plt.subplots(2, 2)
    mpl.style.use("ggplot")  # type: ignore[reportAttributeAccessIssue]

    # Metric: Optimization runtime
    mean_data = (
        statistics_data[statistics_data["Metric"] == Metrics.RUNTIME_MEAN.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )

    std_data = (
        statistics_data[statistics_data["Metric"] == Metrics.RUNTIME_STD.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )
    std_data_asymmetric = np.array(
        [[np.zeros(std_data[col].shape), std_data[col].to_numpy()] for col in std_data]
    )

    axs[0, 0].grid(True, zorder=0)
    mean_data.plot(
        kind="bar",
        yerr=std_data_asymmetric,
        ax=axs[0, 0],
        legend=False,
        rot=45,
        zorder=3,
    )
    axs[0, 0].set_yscale("log", nonpositive="clip")
    axs[0, 0].set_ylabel(DisplayMetrics.RUNTIME.value, fontsize=10)

    # Metric: No. function evaluations
    mean_data = (
        statistics_data[statistics_data["Metric"] == Metrics.NFEV_MEAN.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )

    std_data = (
        statistics_data[statistics_data["Metric"] == Metrics.NFEV_STD.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )
    std_data_asymmetric = np.array(
        [[np.zeros(std_data[col].shape), std_data[col].to_numpy()] for col in std_data]
    )

    axs[0, 1].grid(True, zorder=0)
    mean_data.plot(
        kind="bar",
        yerr=std_data_asymmetric,
        ax=axs[0, 1],
        legend=False,
        rot=45,
        zorder=3,
    )
    axs[0, 1].set_yscale("log", nonpositive="clip")
    axs[0, 1].set_ylabel(DisplayMetrics.NFEV.value, fontsize=10)

    # Metric: Finding rate of global minimum
    mean_data = (
        statistics_data[statistics_data["Metric"] == Metrics.SUCCESS.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )

    axs[1, 0].grid(True, zorder=0)
    mean_data.plot(kind="bar", ax=axs[1, 0], legend=False, rot=45, zorder=3)
    axs[1, 0].set_ylabel(DisplayMetrics.SUCCESS.value, fontsize=10)

    # Metric: Minimum function value error
    mean_data = (
        statistics_data[statistics_data["Metric"] == Metrics.AE_F_MEAN.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )

    std_data = (
        statistics_data[statistics_data["Metric"] == Metrics.AE_F_STD.value]
        .drop(columns=["Metric"])
        .set_index("Problem")
    )
    std_data_asymmetric = np.array(
        [[np.zeros(std_data[col].shape), std_data[col].to_numpy()] for col in std_data]
    )

    axs[1, 1].grid(True, zorder=0)
    mean_data.plot(
        kind="bar",
        yerr=std_data_asymmetric,
        ax=axs[1, 1],
        legend=False,
        rot=45,
        zorder=3,
    )
    axs[1, 1].set_yscale("log", nonpositive="clip")
    axs[1, 1].set_ylabel(DisplayMetrics.MAE_F.value, fontsize=10)

    # Finalize metrics plot
    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.suptitle("Optimizer metrics", fontsize=14)
    fig.set_size_inches(25.6, 14.4)
    plt.savefig(
        Path(SAVE_DIR) / f"optimizer_statistics_{ndims}D.pdf", bbox_inches="tight"
    )
    fig_manager = plt.get_current_fig_manager()
    if fig_manager is not None:
        fig_manager.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s",
    )
    nruns = 100
    testproblems_2d = [
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
        tps.Schaffer2(),
        tps.Schaffer4(),
        tps.StyblinskiTang(2),
    ]

    testproblems_15d = [
        tps.Rastrigin(15),
        tps.Sphere(15),
        tps.Rosenbrock(15),
        tps.StyblinskiTang(15),
    ]

    metrics_2d = calculate_optimizer_metrics(testproblems_2d, nruns, plot_traces=True)
    metrics_15d = calculate_optimizer_metrics(
        testproblems_15d, nruns, plot_traces=False
    )

    statistics_2d = collect_statistics(testproblems_2d, metrics_2d, 2)
    statistics_15d = collect_statistics(testproblems_15d, metrics_15d, 15)

    plot_statistics(statistics_2d, 2)
    plot_statistics(statistics_15d, 15)
