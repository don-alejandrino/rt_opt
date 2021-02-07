import itertools
import time

import dlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import pandas as pd
from scipy.optimize import differential_evolution, shgo, dual_annealing
from tqdm import tqdm

from rt_optimizer.optimizer import run_and_tumble
from rt_optimizer.testproblems import *


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


class lipoWrapper:
    def __init__(self, fun):
        self.__fun = fun

    def f(self, *args):
        return self.__fun(np.array(args))


if __name__ == '__main__':
    n_runs = 200
    testproblems_2D = {'Ackley': Ackley(),
                       'Beale': Beale(),
                       'GoldsteinPrice': GoldsteinPrice(),
                       'Booth': Booth(),
                       'Bukin6': Bukin6(),
                       'Matyas': Matyas(),
                       'Levi13': Levi13(),
                       'Himmelblau': Himmelblau(),
                       'ThreeHumpCamel': ThreeHumpCamel(),
                       'Easom': Easom(),
                       'CrossInTray': CrossInTray(),
                       'Eggholder': Eggholder(),
                       'Hoelder': Hoelder(),
                       'McCormick': McCormick(),
                       'Schaffer2': Schaffer2(),
                       'Schaffer4': Schaffer4()}

    fig, axs = plt.subplots(4, 4)
    n_total_steps = 16 * 5 * n_runs  # No. problems * No. algorithms * No. runs
    metrics_2D = {'Run-and-Tumble': {},
                  'Differential Evolution': {},
                  'SHGO': {},
                  'Dual Annealing': {},
                  'LIPO': {}
                  }

    print('Collecting optimizer statistics...')
    with tqdm(total=n_total_steps) as pbar:
        for n, (name, problem) in enumerate(testproblems_2D.items()):

            # Run-and-tumble algorithm
            x0 = np.array([problem.bounds.lower, problem.bounds.upper])
            bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            metrics_2D['Run-and-Tumble'][name] = {'runtime': [], 'nfev': [], 'x': [], 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = run_and_tumble(problem.f, x0, bounds=bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                metrics_2D['Run-and-Tumble'][name]['runtime'].append(runtime)
                metrics_2D['Run-and-Tumble'][name]['nfev'].append(ret.nfev)
                metrics_2D['Run-and-Tumble'][name]['x'].append(ret.x)
                metrics_2D['Run-and-Tumble'][name]['f'].append(ret.fun)

                if m == 0:  # Plot bacteria traces only once
                    plot_col = n // 4
                    plot_row = n % 4
                    X, Y, Z = gridmap2d(problem.f,
                                        (problem.bounds.lower[0],
                                         problem.bounds.upper[0], 100),
                                        (problem.bounds.lower[1],
                                         problem.bounds.upper[1], 100))
                    cp = axs[plot_row, plot_col].contourf(X, Y, Z, levels=20)
                    axs[plot_row, plot_col].set_xlim([problem.bounds.lower[0],
                                                      problem.bounds.upper[0]])
                    axs[plot_row, plot_col].set_ylim([problem.bounds.lower[1],
                                                      problem.bounds.upper[1]])
                    fig.colorbar(cp, ax=axs[plot_row, plot_col])
                    for single_trace in ret.trace.transpose(1, 0, 2):
                        axs[plot_row, plot_col].plot(single_trace[:, 0], single_trace[:, 1], 'o',
                                                     c='white', ms=0.4)
                    axs[plot_row, plot_col].set_xlabel('x')
                    axs[plot_row, plot_col].set_ylabel('y')
                    axs[plot_row, plot_col].set_title(f'Problem: {name}', fontsize=10)

            # Differential Evolution algorithm
            bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            metrics_2D['Differential Evolution'][name] = {'runtime': [], 'nfev': [], 'x': [],
                                                          'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = differential_evolution(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                metrics_2D['Differential Evolution'][name]['runtime'].append(runtime)
                metrics_2D['Differential Evolution'][name]['nfev'].append(ret.nfev)
                metrics_2D['Differential Evolution'][name]['x'].append(ret.x)
                metrics_2D['Differential Evolution'][name]['f'].append(ret.fun)

            # SHGO algorithm
            bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            metrics_2D['SHGO'][name] = {'runtime': [], 'nfev': [], 'x': [], 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = shgo(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                metrics_2D['SHGO'][name]['runtime'].append(runtime)
                metrics_2D['SHGO'][name]['nfev'].append(ret.nfev)
                metrics_2D['SHGO'][name]['x'].append(ret.x)
                metrics_2D['SHGO'][name]['f'].append(ret.fun)

            # Dual Annealing algorithm
            bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            metrics_2D['Dual Annealing'][name] = {'runtime': [], 'nfev': [], 'x': [], 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret = dual_annealing(problem.f, bounds)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                metrics_2D['Dual Annealing'][name]['runtime'].append(runtime)
                metrics_2D['Dual Annealing'][name]['nfev'].append(ret.nfev)
                metrics_2D['Dual Annealing'][name]['x'].append(ret.x)
                metrics_2D['Dual Annealing'][name]['f'].append(ret.fun)

            # LIPO algorithm
            nfev = 1000
            bounds_lower = problem.bounds.lower.tolist()
            bounds_upper = problem.bounds.upper.tolist()
            lipoFun = lipoWrapper(problem.f)
            metrics_2D['LIPO'][name] = {'runtime': [], 'nfev': [], 'x': [], 'f': []}
            for m in range(n_runs):
                start = time.time()
                ret_lp = dlib.find_min_global(lipoFun.f, bounds_lower, bounds_upper, nfev)
                end = time.time()
                runtime = end - start
                pbar.update(1)
                metrics_2D['LIPO'][name]['runtime'].append(runtime)
                metrics_2D['LIPO'][name]['nfev'].append(nfev)
                metrics_2D['LIPO'][name]['x'].append(ret_lp[0])
                metrics_2D['LIPO'][name]['f'].append(ret_lp[1])

    # Finalize bacteria traces plot
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.suptitle('Run-and-tumble bacteria traces', fontsize=14)
    # ToDo: Save figure
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()

    # Calculate statistics
    metric_names = ['Running time (mean) [s]',
                    'Running time (std) [s]',
                    'No. objective function evaluations (mean)',
                    'No. objective function evaluations (std)',
                    'RMSE in minimum position',
                    'Absolute error in minimum value (mean)',
                    'Absolute error in minimum value (std)']
    display_metrics = ['Running time (s)',
                       'No. objective function evaluations',
                       'RMSE in minimum position',
                       'Absolute error in minimum value']
    algo_names = list(metrics_2D.keys())
    problem_names = list(testproblems_2D.keys())

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
    table_data['Metric'] = display_metrics * len(testproblems_2D)

    for algo, problem in metrics_2D.items():
        for n, (problem_name, metrics) in enumerate(problem.items()):
            mean_runtime = np.array(metrics['runtime']).mean()
            std_runtime = np.array(metrics['runtime']).std()
            mean_nfev = np.array(metrics['nfev']).mean()
            std_nfev = np.array(metrics['nfev']).std()

            true_min_pos = testproblems_2D[problem_name].min.x
            if isinstance(true_min_pos, tuple):  # More than one global minima
                distances = np.min(np.square(np.array(metrics['x'])[:, None, :] -
                                             np.array(true_min_pos)[None, :, :]).sum(axis=2),
                                   axis=1)
                RMSE_x = np.sqrt(distances.mean())
            else:
                RMSE_x = np.sqrt(np.square(np.array(metrics['x']) - true_min_pos).sum(axis=1)
                                 .mean())

            true_min_val = testproblems_2D[problem_name].min.f
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
                MAE_f = np.abs(np.array(metrics['f']) - true_min_val).mean()
                STDAE_f = np.abs(np.array(metrics['f']) - true_min_val).std()

            statistics_data.loc[len(metric_names) * n, algo] = mean_runtime
            statistics_data.loc[len(metric_names) * n + 1, algo] = std_runtime
            statistics_data.loc[len(metric_names) * n + 2, algo] = mean_nfev
            statistics_data.loc[len(metric_names) * n + 3, algo] = std_nfev
            statistics_data.loc[len(metric_names) * n + 4, algo] = RMSE_x
            statistics_data.loc[len(metric_names) * n + 5, algo] = MAE_f
            statistics_data.loc[len(metric_names) * n + 6, algo] = STDAE_f

            table_data.loc[len(display_metrics) * n, algo] = '{0:g} ± {1:g}'.format(mean_runtime,
                                                                                    std_runtime)
            table_data.loc[len(display_metrics) * n + 1, algo] = '{0:g} ± {1:g}'.format(mean_nfev,
                                                                                        std_nfev)
            table_data.loc[len(display_metrics) * n + 2, algo] = '{0:g}'.format(RMSE_x)
            table_data.loc[len(display_metrics) * n + 3, algo] = '{0:g} ± {1:g}'.format(MAE_f,
                                                                                        STDAE_f)

    # Export table data
    html_table = table_data.sort_values(by=['Metric']).reset_index(drop=True).to_html()
    with open('optimizer_statistics.html', 'w', encoding='utf-8') as file:
        file.writelines('<meta charset="UTF-8">\n')
        file.write(html_table)

    # Plot metrics
    fig, axs = plt.subplots(2, 2)
    mpl.style.use('ggplot')

    # Runtime metrics
    mean_data = statistics_data[
        statistics_data['Metric'] == 'Running time (mean) [s]'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'Running time (std) [s]'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    mean_data.plot(kind='bar', yerr=std_data_asymmetric, log=True, ax=axs[0, 0],
                   legend=False, rot=45)
    axs[0, 0].set_ylabel('Running time (s)', fontsize=10)

    # Function evaluation metrics
    mean_data = statistics_data[
        statistics_data['Metric'] == 'No. objective function evaluations (mean)'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'No. objective function evaluations (std)'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    mean_data.plot(kind='bar', yerr=std_data_asymmetric, log=True, ax=axs[0, 1],
                   legend=False, rot=45)
    axs[0, 1].set_ylabel('No. objective function evaluations', fontsize=10)

    # Minimum position error metrics
    mean_data = statistics_data[
        statistics_data['Metric'] == 'RMSE in minimum position'
    ].drop(columns=['Metric']).set_index('Problem')

    mean_data.plot(kind='bar', log=True, ax=axs[1, 0], legend=False, rot=45)
    axs[1, 0].set_ylabel('RMSE in minimum position', fontsize=10)

    # Minimum function value error metrics
    mean_data = statistics_data[
        statistics_data['Metric'] == 'Absolute error in minimum value (mean)'
    ].drop(columns=['Metric']).set_index('Problem')

    std_data = statistics_data[
        statistics_data['Metric'] == 'Absolute error in minimum value (std)'
    ].drop(columns=['Metric']).set_index('Problem')
    std_data_asymmetric = []
    for col in std_data:
        std_data_asymmetric.append([np.zeros(std_data[col].shape), std_data[col].values])

    mean_data.plot(kind='bar', yerr=std_data_asymmetric, log=True, ax=axs[1, 1],
                   legend=False, rot=45)
    axs[1, 1].set_ylabel('Absolute error in minimum value', fontsize=10)

    # Finalize metrics plot
    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.suptitle('Optimizer metrics', fontsize=14)
    # ToDo: Save figure
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()

    # ToDo: 10- or 100-dimensional test functions
