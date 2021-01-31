import time

import dlib
import matplotlib.pyplot as plt
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
    n_dim = 2
    n_runs = 100
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
            if problem.bounds.lower is not None and problem.bounds.upper is not None:
                x0 = np.array([problem.bounds.lower, problem.bounds.upper])
                bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            else:
                x0 = np.array([[-10, -10], [10, 10]])
                bounds = None

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
            if problem.bounds.lower is not None and problem.bounds.upper is not None:
                bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            else:
                bounds = None
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
            if problem.bounds.lower is not None and problem.bounds.upper is not None:
                bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            else:
                bounds = None
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
            if problem.bounds.lower is not None and problem.bounds.upper is not None:
                bounds = np.vstack((problem.bounds.lower, problem.bounds.upper)).T
            else:
                bounds = None
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
            if problem.bounds.lower is not None:
                bounds_lower = problem.bounds.lower.tolist()
            else:
                bounds_lower = [-10, -10]
            if problem.bounds.upper is not None:
                bounds_upper = problem.bounds.upper.tolist()
            else:
                bounds_upper = [10, 10]
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

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.suptitle('Run-and-tumble bacteria traces', fontsize=14)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()

    # ToDo: Show statistics
    for algo, problem in metrics_2D.items():
        print(f'{algo} algorithm:')
        print('=================================')
        for problem_name, metrics in problem.items():
            print(f'{problem_name} problem:')
            print('---------------------------------')
            mean_runtime = np.array(metrics['runtime']).mean()
            mean_nfev = np.array(metrics['nfev']).mean()

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
            else:
                MAE_f = np.abs(np.array(metrics['f']) - true_min_val).mean()

            print(f'Mean running time: {mean_runtime}s')
            print(f'Mean number of objective function evaluations: {mean_nfev}')
            print(f'RMSE in minimum position: {RMSE_x}')
            print(f'MAE in minimum value: {MAE_f}\n')
            # ToDo: Show as table (rows: algorithms, columns: problems)
            # ToDo: Plot as bar plot, for each of the four metrics (colors: algorithms,
            #  x-axis: problems)

    # ToDo: 10- or 100-dimensional test functions
