import time

import dlib
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, shgo, dual_annealing

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
    testproblems_2D = ((Ackley(), 'Ackley'),
                       (Beale(), 'Beale'),
                       (GoldsteinPrice(), 'GoldsteinPrice'),
                       (Booth(), 'Booth'),
                       (Bukin6(), 'Bukin6'),
                       (Matyas(), 'Matyas'),
                       (Levi13(), 'Levi13'),
                       (Himmelblau(), 'Himmelblau'),
                       (ThreeHumpCamel(), 'ThreeHumpCamel'),
                       (Easom(), 'Easom'),
                       (CrossInTray(), 'CrossInTray'),
                       (Eggholder(), 'Eggholder'),
                       (Hoelder(), 'Hoelder'),
                       (McCormick(), 'McCormick'),
                       (Schaffer2(), 'Schaffer2'),
                       (Schaffer4(), 'Schaffer4'))

    for problem, name in testproblems_2D:
        print(f'Problem: {name}')
        print('=================================')

        # Run-and-tumble algorithm
        print('Run-and-tumble algorithm:')
        if problem.constraints.lower is not None and problem.constraints.upper is not None:
            x0 = np.array([problem.constraints.lower, problem.constraints.upper])
            bounds = np.vstack((problem.constraints.lower, problem.constraints.upper)).T
        else:
            x0 = np.array([[-10, -10], [10, 10]])
            bounds = None
        start = time.time()
        ret = run_and_tumble(problem.f, x0, bounds=bounds)
        end = time.time()
        runtime = end - start
        print('---------------------------------')
        print(f'Running time: {runtime}s')
        print(f'Number of objective function evaluations: {ret.nfev}')
        print(f'Global minimum: x = {ret.x}, f(x) = {ret.fun}')

        X, Y, Z = gridmap2d(problem.f,
                            (problem.constraints.lower[0], problem.constraints.upper[0], 100),
                            (problem.constraints.lower[1], problem.constraints.upper[1], 100))
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(X, Y, Z, levels=20)
        ax.set_xlim([problem.constraints.lower[0], problem.constraints.upper[0]])
        ax.set_ylim([problem.constraints.lower[1], problem.constraints.upper[1]])
        fig.colorbar(cp)
        for single_trace in ret.trace.transpose(1, 0, 2):
            ax.plot(single_trace[:, 0], single_trace[:, 1], 'o', c='white', ms=0.7)
        plt.show()

        # Differential Evolution algorithm
        print('\nDifferential Evolution algorithm:')
        if problem.constraints.lower is not None and problem.constraints.upper is not None:
            bounds = np.vstack((problem.constraints.lower, problem.constraints.upper)).T
        else:
            bounds = None
        start = time.time()
        ret_bh = differential_evolution(problem.f, bounds)
        end = time.time()
        runtime = end - start
        print('---------------------------------')
        print(f'Running time: {runtime}s')
        print(f'Number of objective function evaluations: {ret_bh.nfev}')
        print(f'Global minimum: x = {ret_bh.x}, f(x) = {ret_bh.fun}')

        # SHGO algorithm
        print('\nSHGO algorithm:')
        if problem.constraints.lower is not None and problem.constraints.upper is not None:
            bounds = np.vstack((problem.constraints.lower, problem.constraints.upper)).T
        else:
            bounds = None
        start = time.time()
        ret_bh = shgo(problem.f, bounds)
        end = time.time()
        runtime = end - start
        print('---------------------------------')
        print(f'Running time: {runtime}s')
        print(f'Number of objective function evaluations: {ret_bh.nlfev}')
        print(f'Global minimum: x = {ret_bh.x}, f(x) = {ret_bh.fun}')

        # Dual Annealing algorithm
        print('\nDual Annealing algorithm:')
        if problem.constraints.lower is not None and problem.constraints.upper is not None:
            bounds = np.vstack((problem.constraints.lower, problem.constraints.upper)).T
        else:
            bounds = None
        start = time.time()
        ret_bh = dual_annealing(problem.f, bounds)
        end = time.time()
        runtime = end - start
        print('---------------------------------')
        print(f'Running time: {runtime}s')
        print(f'Number of objective function evaluations: {ret_bh.nfev}')
        print(f'Global minimum: x = {ret_bh.x}, f(x) = {ret_bh.fun}')

        # LIPO algorithm
        print('\nLIPO algorithm:')
        nfev = 1000
        if problem.constraints.lower is not None:
            bounds_lower = problem.constraints.lower.tolist()
        else:
            bounds_lower = [-10, -10]
        if problem.constraints.upper is not None:
            bounds_upper = problem.constraints.upper.tolist()
        else:
            bounds_upper = [10, 10]
        lipoFun = lipoWrapper(problem.f)
        start = time.time()
        ret_lp = dlib.find_min_global(lipoFun.f, bounds_lower, bounds_upper, nfev)
        end = time.time()
        runtime = end - start
        print('---------------------------------')
        print(f'Running time: {runtime}s')
        print(f'Number of objective function evaluations: {nfev}')
        print(f'Global minimum: x = {ret_lp[0]}, f(x) = {ret_lp[1]}')

        print('\n')

        # ToDo: Run each problem for 100 times and collect statistics
        # ToDo: Save traces and plot all traces in one figure
        # ToDo: 10-dimensional test functions
