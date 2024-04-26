import itertools
import time
from os.path import join

import dlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing
from tqdm import tqdm

from rt_opt.optimizer import optimize
from rt_opt.testproblems_shifted import *


SAVE_DIR = 'demo/results'


def gridmap2d(fun, x_specs, y_specs):
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


def calculate_optimizer_metrics(problems, n_runs, plot_traces=False):
    """
    Let the global optimizers rt_opt, scipy's differential_evolution, scipy's dual_annealing,
    and dlib's LIPO minimize a bunch of test functions and collect performance metrics.

    :param problems: [list<TestProblem instance>] List of test problems to be used
    :param n_runs: [int] How often each problem is solved by the different optimizers
    :param plot_traces: [bool] Whether to plot bacteria traces for the rt_opt
    :return: Performance metrics [dict]
    """

    optimizer_results = {
        'Run-and-Tumble': {},
        'Differential Evolution': {},
        'Dual Annealing': {},
        'LIPO': {}
    }

    if plot_traces:
        n_plots = len(problems)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols)

    n_total_steps = len(problems) * len(optimizer_results) * n_runs

    print('Collecting optimizer statistics...')
    with tqdm(total=n_total_steps) as pbar:
        for n, problem in enumerate(problems):
            name = problem.__class__.__name__
            if problem.bounds.lower is not None:
                bounds_lower = problem.bounds.lower
            else:
                bounds_lower = np.repeat(-5, problem.ndims)
            if problem.bounds.upper is not None:
                bounds_upper = problem.bounds.upper
            else:
                bounds_upper = np.repeat(5, problem.ndims)

            # Run-and-tumble algorithm
            bounds = np.vstack((bounds_lower, bounds_upper)).T
            optimizer_results['Run-and-Tumble'][name] = {'runtime': [],
                                                         'nfev': [],
                                                         'x': [],
                                                         'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = optimize(problem.f, bounds=bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                optimizer_results['Run-and-Tumble'][name]['runtime'].append(runtime)
                optimizer_results['Run-and-Tumble'][name]['nfev'].append(ret.nfev)
                optimizer_results['Run-and-Tumble'][name]['x'].append(ret.x)
                optimizer_results['Run-and-Tumble'][name]['f'].append(ret.fun)

                if plot_traces and m == 0:  # Plot bacteria traces only once
                    if problem.ndims != 2:
                        raise NotImplementedError('Currently, plotting bacteria traces is ' +
                                                  'supported for 2D problems only.')
                    # noinspection PyUnboundLocalVariable
                    plot_row = n // n_cols
                    # noinspection PyUnboundLocalVariable
                    plot_col = n % n_cols
                    X, Y, Z = gridmap2d(problem.f,
                                        (bounds_lower[0], bounds_upper[0], 100),
                                        (bounds_lower[1], bounds_upper[1], 100))
                    # noinspection PyUnboundLocalVariable
                    cp = axs[plot_row, plot_col].contourf(X, Y, Z, levels=20)
                    axs[plot_row, plot_col].set_xlim([bounds_lower[0], bounds_upper[0]])
                    axs[plot_row, plot_col].set_ylim([bounds_lower[1], bounds_upper[1]])
                    # noinspection PyUnboundLocalVariable
                    fig.colorbar(cp, ax=axs[plot_row, plot_col])
                    for single_trace in ret.trace.transpose(1, 0, 2):
                        axs[plot_row, plot_col].plot(single_trace[:, 0], single_trace[:, 1], 'o',
                                                     c='white', ms=0.4)
                    axs[plot_row, plot_col].set_xlabel('x')
                    axs[plot_row, plot_col].set_ylabel('y')
                    axs[plot_row, plot_col].set_title(f'Problem: {name}', fontsize=10)

            # Differential Evolution algorithm
            bounds = np.vstack((bounds_lower, bounds_upper)).T
            optimizer_results['Differential Evolution'][name] = {'runtime': [],
                                                                 'nfev': [],
                                                                 'x': [],
                                                                 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = differential_evolution(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                optimizer_results['Differential Evolution'][name]['runtime'].append(runtime)
                optimizer_results['Differential Evolution'][name]['nfev'].append(ret.nfev)
                optimizer_results['Differential Evolution'][name]['x'].append(ret.x)
                optimizer_results['Differential Evolution'][name]['f'].append(ret.fun)

            # Dual Annealing algorithm
            bounds = np.vstack((bounds_lower, bounds_upper)).T
            optimizer_results['Dual Annealing'][name] = {'runtime': [],
                                                         'nfev': [],
                                                         'x': [],
                                                         'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = dual_annealing(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                optimizer_results['Dual Annealing'][name]['runtime'].append(runtime)
                optimizer_results['Dual Annealing'][name]['nfev'].append(ret.nfev)
                optimizer_results['Dual Annealing'][name]['x'].append(ret.x)
                optimizer_results['Dual Annealing'][name]['f'].append(ret.fun)

            # LIPO algorithm
            nfev = 500 * problem.ndims
            lipoFun = LipoWrapper(problem.f)
            optimizer_results['LIPO'][name] = {'runtime': [], 'nfev': [], 'x': [], 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret_lp = dlib.find_min_global(lipoFun.f, bounds_lower.tolist(),
                                              bounds_upper.tolist(), nfev)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                optimizer_results['LIPO'][name]['runtime'].append(runtime)
                optimizer_results['LIPO'][name]['nfev'].append(nfev)
                optimizer_results['LIPO'][name]['x'].append(ret_lp[0])
                optimizer_results['LIPO'][name]['f'].append(ret_lp[1])

    if plot_traces:
        # Finalize bacteria traces plot
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        fig.suptitle('Run-and-tumble bacteria traces', fontsize=14)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        fig.set_size_inches(25.6, 14.4)
        plt.savefig(join(SAVE_DIR, 'bacteria_traces.png'), bbox_inches='tight', dpi=150)
        plt.show()

    return optimizer_results


def show_statistics(problems, optimizer_results, ndims):
    """
    Calculate and display optimizer performance statistics.

    :param problems: [list<TestProblem instance>] List of test problems to be used
    :param optimizer_results: [dict] Optimizer performance metrics
    :param ndims: [int] Dimension of the test problems
    """

    metric_names = ['Running time (mean) [s]',
                    'Running time (std) [s]',
                    'No. objective function evaluations (mean)',
                    'No. objective function evaluations (std)',
                    'RMSE in minimum position',
                    'Absolute error in minimum value (mean)',
                    'Absolute error in minimum value (std)',
                    'Finding rate of global minimum']
    display_metrics = ['Running time (s)',
                       'No. objective function evaluations',
                       'RMSE in minimum position',
                       'Absolute error in minimum value',
                       'Finding rate of global minimum']
    algo_names = list(optimizer_results.keys())
    problems_dict = {prob.__class__.__name__: prob for prob in problems}
    problem_names = list(problems_dict.keys())

    statistics_data = pd.DataFrame(columns=['Problem', 'Metric'] + algo_names)
    statistics_data['Problem'] = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, len(metric_names)) for x in problem_names
        )
    )
    statistics_data['Metric'] = metric_names * len(problem_names)

    table_data = pd.DataFrame(columns=['Problem', 'Metric'] + algo_names)
    table_data['Problem'] = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, len(display_metrics)) for x in problem_names
        )
    )
    table_data['Metric'] = display_metrics * len(problems)

    for algo, results in optimizer_results.items():
        for n, (problem_name, metrics) in enumerate(results.items()):
            mean_runtime = np.array(metrics['runtime']).mean()
            std_runtime = np.array(metrics['runtime']).std()
            mean_nfev = np.array(metrics['nfev']).mean()
            std_nfev = np.array(metrics['nfev']).std()

            true_min_pos = problems_dict[problem_name].min.x
            if isinstance(true_min_pos, tuple):  # More than one global minima
                distances = np.min(np.square(np.array(metrics['x'])[:, None, :] -
                                             np.array(true_min_pos)[None, :, :]).sum(axis=2),
                                   axis=1)
                RMSE_x = np.sqrt(distances.mean())
            else:
                RMSE_x = np.sqrt(np.square(np.array(metrics['x']) - true_min_pos).sum(axis=1)
                                 .mean())

            true_min_val = problems_dict[problem_name].min.f
            if isinstance(true_min_val, tuple):  # Only range for minimum value known
                errors = np.empty(len(metrics['f']))
                for j, val in enumerate(metrics['f']):
                    if val < true_min_val[0]:
                        errors[j] = val - true_min_val[0]
                    elif val > true_min_val[1]:
                        errors[j] = val - true_min_val[1]
                    else:
                        errors[j] = 0
                MAE_f = np.abs(errors).mean()
                STDAE_f = np.abs(errors).std()
            else:
                errors = np.array(metrics['f']) - true_min_val
                MAE_f = np.abs(errors).mean()
                STDAE_f = np.abs(errors).std()

            min_found = np.where(np.abs(errors) < 1e-9, 1, 0)
            finding_rate = min_found.sum() / len(min_found)

            statistics_data.loc[len(metric_names) * n, algo] = mean_runtime
            statistics_data.loc[len(metric_names) * n + 1, algo] = std_runtime
            statistics_data.loc[len(metric_names) * n + 2, algo] = mean_nfev
            statistics_data.loc[len(metric_names) * n + 3, algo] = std_nfev
            statistics_data.loc[len(metric_names) * n + 4, algo] = RMSE_x
            statistics_data.loc[len(metric_names) * n + 5, algo] = MAE_f
            statistics_data.loc[len(metric_names) * n + 6, algo] = STDAE_f
            statistics_data.loc[len(metric_names) * n + 7, algo] = finding_rate

            table_data.loc[len(display_metrics) * n, algo] = '{0:g} ± {1:g}'.format(mean_runtime,
                                                                                    std_runtime)
            table_data.loc[len(display_metrics) * n + 1, algo] = '{0:g} ± {1:g}'.format(mean_nfev,
                                                                                        std_nfev)
            table_data.loc[len(display_metrics) * n + 2, algo] = '{0:g}'.format(RMSE_x)
            table_data.loc[len(display_metrics) * n + 3, algo] = '{0:g} ± {1:g}'.format(MAE_f,
                                                                                        STDAE_f)
            table_data.loc[len(display_metrics) * n + 4, algo] = '{0:.3f}'.format(finding_rate)

    # Export table data
    html_table = table_data.sort_values(by=['Metric', 'Problem']).reset_index(drop=True).to_html()
    with open(join(SAVE_DIR, f'optimizer_statistics_{ndims}D.html'), 'w', encoding='utf-8') as file:
        file.writelines('<meta charset="UTF-8">\n')
        file.write(html_table)

    # Plot metrics
    fig, axs = plt.subplots(2, 2)
    mpl.style.use('ggplot')

    # Metric: Runtime
    mean_data = statistics_data[
        statistics_data['Metric'] == 'Running time (mean) [s]'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'Running time (std) [s]'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    axs[0, 0].grid(True, zorder=0)
    mean_data.plot(kind='bar', yerr=std_data_asymmetric, ax=axs[0, 0], legend=False, rot=45,
                   zorder=3)
    axs[0, 0].set_yscale("log", nonpositive="clip")
    axs[0, 0].set_ylabel('Running time (s)', fontsize=10)

    # Metric: No. function evaluations
    mean_data = statistics_data[
        statistics_data['Metric'] == 'No. objective function evaluations (mean)'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'No. objective function evaluations (std)'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    axs[0, 1].grid(True, zorder=0)
    mean_data.plot(kind='bar', yerr=std_data_asymmetric, ax=axs[0, 1], legend=False, rot=45,
                   zorder=3)
    axs[0, 1].set_yscale("log", nonpositive="clip")
    axs[0, 1].set_ylabel('No. objective function evaluations', fontsize=10)

    # Metric: Finding rate of global minimum
    mean_data = statistics_data[
        statistics_data['Metric'] == 'Finding rate of global minimum'
    ].drop(columns=['Metric']).set_index('Problem')

    axs[1, 0].grid(True, zorder=0)
    mean_data.plot(kind='bar', ax=axs[1, 0], legend=False, rot=45, zorder=3)
    axs[1, 0].set_ylabel('Finding rate of global minimum', fontsize=10)

    # Metric: Minimum function value error
    mean_data = statistics_data[
        statistics_data['Metric'] == 'Absolute error in minimum value (mean)'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'Absolute error in minimum value (std)'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    axs[1, 1].grid(True, zorder=0)
    mean_data.plot(kind='bar', yerr=std_data_asymmetric, ax=axs[1, 1], legend=False, rot=45,
                   zorder=3)
    axs[1, 1].set_yscale("log", nonpositive="clip")
    axs[1, 1].set_ylabel('Absolute error in minimum value', fontsize=10)

    # Finalize metrics plot
    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.suptitle('Optimizer metrics', fontsize=14)
    fig.set_size_inches(25.6, 14.4)
    plt.savefig(join(SAVE_DIR, f'optimizer_statistics_{ndims}D.pdf'), bbox_inches='tight')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.full_screen_toggle()
    plt.show()


class LipoWrapper:
    def __init__(self, fun):
        self.__fun = fun

    def f(self, *args):
        return self.__fun(np.array(args))


if __name__ == '__main__':
    nruns = 100
    testproblems_2D = [
        Rastrigin(2),
        Ackley(),
        Sphere(2),
        Rosenbrock(2),
        Beale(),
        GoldsteinPrice(),
        Booth(),
        Bukin6(),
        Matyas(),
        Levi13(),
        Himmelblau(),
        ThreeHumpCamel(),
        Easom(),
        CrossInTray(),
        Eggholder(),
        Hoelder(),
        McCormick(),
        Schaffer2(),
        Schaffer4(),
        StyblinskiTang(2)
    ]

    testproblems_15D = [
        Rastrigin(15),
        Sphere(15),
        Rosenbrock(15),
        StyblinskiTang(15)
    ]

    metrics_2D = calculate_optimizer_metrics(testproblems_2D, nruns, plot_traces=True)
    metrics_15D = calculate_optimizer_metrics(testproblems_15D, nruns, plot_traces=False)

    show_statistics(testproblems_2D, metrics_2D, 2)
    show_statistics(testproblems_15D, metrics_15D, 15)
