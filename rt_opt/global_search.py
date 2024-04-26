import numpy as np
from scipy.stats import stats


def run_and_tumble(f, x0_population, projection_callback, niter, stepsize_start, stepsize_end,
                   base_tumble_rate=0.1, stationarity_window=20, eps_stat=1e-3, attraction=False,
                   attraction_window=10, attraction_sigma=1, attraction_strength=0.5,
                   bounds_reflection=False, verbosity=1):
    """
    Implementation of a bacterial run-and-tumble optimizer algorithm, motivated by the chemotactic
    behavior of E.Coli. The motion of E.Coli consists of directed, ballistic "runs", interrupted by
    sudden random re-orientations, so-called "tumbles", that appear at some given rate. If a
    bacterium detects to swim toward higher concentrations of an attractant (i.e., if the attractant
    concentration increases during a run), its tumbling rate is lowered, thus inducing an effective
    movement toward the attractant's source.
    Here, we implement a simplified E.Coli chemotaxis model, where the attractant concentration is
    the negative of a given objective function.

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x0_population: [np.array] Initial condition for the bacteria population.
           x0_population.shape[0] defines the number of bacteria and x0_population.shape[1] the
           problem dimensionality
    :param projection_callback: [callable] Bounds projection, see description of parameter
           ``projection_callback`` in :func:`local_search.bfgs_b`
    :param niter: [int] Maximum number of run-and-tumble steps
    :param stepsize_start: [float] Defines the initial length of a "run" step
    :param stepsize_end: [float] Defines the final length of a "run" step at the last iteration.
           The actual stepsize decreases quadratically from stepsize_start to stepsize_end
    :param base_tumble_rate: [float] "Undisturbed" tumble rate when a bacterium does not feel any
           change in attractant concentration
    :param stationarity_window: [int] If the mean position of all bacteria has had a relative change
           less than eps_stat over a step window stationarity_window, the bacteria distribution is
           considered to be stationary and the algorithm stops
    :param eps_stat: [float] See description of parameter ``stationarity_window``
    :param attraction: [bool] Whether the bacteria attract each other or not. We model bacteria
           attraction the following way: Each bacterium is supposed to leave some kind of magic
           attractant at the places it has visited thus far, that attracts all other bacteria
    :param attraction_window: [int] Defines the number of recent positions in a bacterium's trace
           that contributes to the attraction mechanism. We have to define this cut-off length,
           since otherwise calculating the bacteria attractions becomes computationally very
           expensive. This parameter only has an effect if attraction == True
    :param attraction_sigma: [float] The bacterial attractant concentration is modeled to decay
           according to a Gaussian distribution,
           -----
           attraction_strength / 2 / attraction_sigma ** 2
                * exp(-(np.square(x - x_vis) / 2 / attraction_sigma ** 2)),
           -----
           around each point x_vis visited thus far. This parameter only has an effect if
           attraction == True
    :param attraction_strength: [float] See description of parameter ``attraction_sigma``. Note that
           if attraction_strength < 0, the bacterial attraction turns into a repulsion. This
           parameter only has an effect if attraction == True
    :param bounds_reflection: [bool] Whether bacteria reverse their direction when hitting a
           boundary (True) or tumble randomly (False)
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :return: (x_best, f_best, nfev, nit, trace), where
             - x_best [np.array] is the best x found so far,
             - f_best [float] is the corresponding objective function value,
             - nfev [int] is the number of objective function evaluations taken,
             - nit [int] is the number of run-and-tumble iterations, and
             - trace [np.array] is the bacteria population trace, i.e., contains all visited points
               of x for each bacterium
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'

    n_bacteria = x0_population.shape[0]
    n_dims = x0_population.shape[1]

    x = x0_population.copy()
    x_best = x0_population.copy()
    x_old = x0_population.copy()
    f_old = np.array([f(val) for val in x_old])
    f_best = f_old.copy()
    f_max = f_old.copy()
    nfev = n_bacteria
    trace = np.empty((niter + 1, n_bacteria, n_dims))
    trace[0] = x0_population.copy()
    x_sum = x0_population.sum(axis=0)
    x_mean_history = []

    # Initial random bacteria orientations
    v = np.empty(x0_population.shape)
    for m in range(n_bacteria):
        v_m = np.random.uniform(-1, 1, n_dims)
        while not v_m.any():
            v_m = np.random.uniform(-1, 1, n_dims)
        v_m = v_m / np.sqrt(np.sum(v_m ** 2))
        v[m] = v_m

    for n in range(niter):
        alpha = stepsize_start + (stepsize_end - stepsize_start) * (n ** 2) / (niter ** 2)

        if attraction:
            # Calculate attraction between the bacteria traces
            kernel = (x[:, None, None, :] -
                      trace[None, (n + 1 - min(n, attraction_window)):(n + 1), :, :])
            grad_attractant = (
                    attraction_strength / 2 / (attraction_sigma ** 2) * kernel *
                    np.exp(-(np.square(kernel) / 2 /
                             (attraction_sigma ** 2)).sum(axis=3))[:, :, :, None]
            ).sum(axis=(1, 2))
        else:
            grad_attractant = np.zeros(x.shape)

        # Run
        x = x_old + (v - grad_attractant) * alpha
        x, bounds_hit = projection_callback(x)
        f_new = np.array([f(val) for val in x])
        nfev += n_bacteria
        trace[n + 1] = x.copy()

        # Tumble
        # We add a small constant (1e-9) to the denominator below, in order to account for the fact
        # that a bacterium may be stuck at the boundary, in which case x[i, :] = x_old[i, :]
        delta_f = (f_new - f_old) / (np.sqrt(np.sum((x - x_old) ** 2, axis=1)) + 1e-9)

        # Avoid exp over/underflow
        delta_f = np.maximum(np.minimum(delta_f, 100), -100)
        tumble_rate = base_tumble_rate * np.exp(delta_f)

        # Calculate new orientation
        for m, tr in enumerate(tumble_rate):
            if bounds_reflection and bounds_hit[m].any():
                # Reflection at boundaries
                v[m] = -v[m]
            elif bounds_hit[m].any() or np.random.uniform() > 1 - tr:
                # Realistically, tr must be clipped to [0, 1]. However, the inequality above is not
                # influenced by this clipping, and we thus omit it in order to save computation time
                v_m = np.random.uniform(-1, 1, n_dims)
                while not v_m.any():
                    v_m = np.random.uniform(-1, 1, n_dims)
                v_m = v_m / np.sqrt(np.sum(v_m ** 2))
                v[m] = v_m

        # Remember best results
        x_best = np.where((f_new < f_best)[:, None], x, x_best)
        f_best = np.minimum(f_new, f_best)
        f_max = np.maximum(f_new, f_max)
        x_old = x.copy()
        f_old = f_new.copy()

        if verbosity == 2:
            for m in range(n_bacteria):
                print('Run-and-tumble step {}, bacterium {}:\tx = {}, f(x) = {}'
                      .format(n + 1, m, x[m], f_new[m]))

        # Calculate mean position of the bacteria
        x_sum = x_sum + x.sum(axis=0)
        x_mean = x_sum / n_bacteria / (n + 1)
        x_mean_history.append(x_mean)
        if (n + 1) % stationarity_window == 0:
            # If the mean position has had a relative change less than eps_stat over a step window
            # stationarity_window, we consider the bacteria distribution as stationary
            window = np.array(x_mean_history[-stationarity_window:]).sum(axis=1)
            slope, intercept, rValue, _pValue, _stdErr = stats.linregress(
                np.linspace(0, 1, len(window)), window
            )
            if rValue ** 2 > 0.9 and abs(slope / intercept) < eps_stat:
                nit = n + 1
                if verbosity > 0:
                    print(f'Run-and-tumble stage: Stationary state detected after {nit} steps.')
                break

    else:
        if verbosity > 0:
            print('Run-and-tumble stage: No stationary state could be detected after ' +
                  f'{niter + 1} iterations. Please try increasing niter or the stationarity ' +
                  f'detection threshold eps_stat.')
        nit = niter + 1

    trace = trace[:(nit + 1)]

    return x_best, f_best, nfev, nit, trace
