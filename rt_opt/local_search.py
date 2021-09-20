from types import SimpleNamespace

import numpy as np


def bfgs_b(f, x0, projection_callback, H_start=None, a=1, c=1e-6, niter=100, n_linesearch=20,
           alpha_linesearch=0.5, beta_linesearch=0.5, eps_abs=1e-9, eps_rel=1e-6, verbosity=1):
    """
    Implementation of the BFGS algorithm for arbitrarily bounded search regions. An estimate of the
    optimal step size for each iteration is found using a two-way-backtracking line search
    algorithm.

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x0: [np.array] Initial condition
    :param projection_callback: [callable] Bounds projection. The function bounds(x) must return a
           tuple (x_projected, bounds_hit), where x_projected is the input variable x projected to
           the defined search region. That is, if x is within this region, it is returned unchanged,
           whereas if it is outside this region, it is projected to the region's boundaries. The
           second output, bounds_hit, indicates whether the boundary has been hit for each component
           of x. If, for example, x is three-dimensional and has hit the search region's boundaries
           in x_1 and x_2, but not in x_3, bounds_hit = [True, True, False]. Note that the search
           domain needs not necessarily be rectangular. Therefore, we define a "boundary hit" in any
           component of x in the following way:
           bounds_hit[i] = True iff either x + δê_i or x - δê_i is outside the defined search
           domain ∀ δ ∈ ℝ⁺, where ê_i is the i-th unit vector
    :param H_start: [np.array] Initial Hessian at x0
    :param a: [float] Initial line search step size
    :param c: [float] Numerical differentiation step size
    :param niter: [int] Maximum number of BFGS iterations
    :param n_linesearch: [int] Maximum number of linesearch steps in each iteration
    :param alpha_linesearch: [float] Line search control parameter alpha, see description in
           :func:`two_way_linesearch`. Must be in between 0 and 1
    :param beta_linesearch: [float] Line search control parameter beta, see description in
           :func:`two_way_linesearch`. Must be in between 0 and 1
    :param eps_abs: [float] Absolute tolerance
    :param eps_rel: [float] Relative tolerance
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :return: (x_best, f_best, nfev, nit, success, trace), where
             - x_best [np.array] is the best x found so far,
             - f_best [float] is the corresponding objective function value,
             - nfev [int] is the number of objective function evaluations taken,
             - nit [int] is the number of BFGS iterations,
             - success [bool] indicates whether the BGFS algorithm finished successfully, i.e,
               whether absolute and relative tolerances were met, and
             - trace [np.array] is the optimizer trace, i.e., contains all visited points of x
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'

    def calculate_gradient():
        gradient = np.zeros(n_dims)
        for m in range(n_dims):
            unit_vec = np.zeros(n_dims)
            unit_vec[m] = 1
            f_minus = f(x - c * unit_vec)
            f_plus = f(x + c * unit_vec)
            gradient[m] = (f_plus - f_minus) / 2 / c

        return gradient

    n_dims = len(x0)
    trace = np.empty((niter, n_dims))

    if H_start is None or np.max(np.abs(H_start)) < 1e-3:
        B = np.identity(n_dims)
    else:
        assert (len(H_start.shape) == 2 and
                H_start.shape[0] == n_dims and
                H_start.shape[1] == n_dims), 'H_start has wrong format.'
        try:
            B = np.linalg.inv(H_start)
        except np.linalg.LinAlgError:
            B = np.identity(n_dims)

    nfev = 0
    x = x0.copy()
    x, _ = projection_callback(x)
    x_best = x.copy()
    grad = calculate_gradient()
    nfev += 2 * n_dims
    f_curr = f_best = f(x)
    nfev += 1
    acc0 = np.linalg.norm(x - projection_callback(x - grad)[0])
    for k in range(niter):
        _, bounds_hit = projection_callback(x)
        B[bounds_hit] = np.identity(n_dims)[bounds_hit]

        # Calculate search direction
        d = -B.dot(grad)

        # Calculate optimal step size and update x
        a_old = a
        ls_result = two_way_linesearch(f, x, grad, d, a, n_linesearch, f_curr, projection_callback,
                                       alpha_linesearch, beta_linesearch)
        f_curr = ls_result.f
        x = ls_result.x
        a = ls_result.a
        nfev += ls_result.nfev

        if not ls_result.success and verbosity == 2:
            print(f"BGFS step {k + 1}: Couldn't find sufficiently good step size during " +
                  f"{n_linesearch} line search steps.")

        if f_curr < f_best:
            f_best = f_curr
            x_best = x.copy()
        else:
            a = a_old * 0.95

        # Update inverse Hessian approximation
        s = a * d
        s[bounds_hit] = 0
        grad_new = calculate_gradient()
        nfev += 2 * n_dims
        y = grad_new - grad
        y[bounds_hit] = 0
        if np.dot(y, s) <= 0:
            B = np.identity(n_dims)
        else:
            B = (
                (np.identity(n_dims) - np.outer(s, y) / np.dot(y, s))
                .dot(B)
                .dot(np.identity(n_dims) - np.outer(y, s) / np.dot(y, s)) +
                np.outer(s, s) / np.dot(y, s)
            )

        grad = grad_new
        trace[k] = x.copy()
        if verbosity == 2:
            print('BGFS step {}:\tx = {}, f(x) = {}'.format(k + 1, x, f_curr))

        acc = np.linalg.norm(x - projection_callback(x - grad)[0])
        if acc <= eps_abs + eps_rel * acc0:
            nit = k + 1
            success = True
            if verbosity > 0:
                print(f'BGFS target accuracy reached after {nit} steps.')
            break

    else:
        if verbosity > 0:
            print(f'Could not reach desired BGFS accuracy after {niter + 1} iterations. Please ' +
                  'try increasing the number of iterations or the tolerance.')
        nit = niter + 1
        success = False

    trace = trace[:nit]

    res = {'x': x_best, 'f': f_best, 'nfev': nfev, 'nit': nit, 'success': success, 'trace': trace}
    return SimpleNamespace(**res)


def two_way_linesearch(f, x, grad, d, a, niter, f_old, projection_callback, alpha, beta):
    """
    Implementation of a two-way-backtracking line search algorithm, as outlined in
    +++++
    T. T. Truong, T. H. Nguyen, Backtracking gradient descent method for general C1 functions, with
    applications to Deep Learning, arXiv:1808.05160 (2018).
    +++++
    Here, we also include a projection of x onto a bounded subspace. The Armijo condition deciding
    whether a stepsize a is accepted reads in this case:
    f(P(x + a * d)) ≤ f(x) - alpha * ∇f(x)•(x - P(x + a * d)),
    where d is the search direction and P the projection of x onto a bounded subspace.

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x: [np.array] Current (starting) position
    :param grad: [np.array] Gradient of f at the current (starting) position x
    :param d: [np.array] Search direction
    :param a: [float] Initial stepsize
    :param niter: [int] Maximum number of line search iterations
    :param f_old: [float] Objective function value at the beginning, f(x)
    :param projection_callback: [callable] Bounds projection, see description of parameter
           ``projection_callback`` in :func:`bfgs_b`
    :param alpha: [float] Line search control parameter alpha. Must be in between 0 and 1
    :param beta: [float] Line search control parameter beta. Must be in between 0 and 1
    :return: Namespace object with the following attributes:
             - success: [bool] Whether the line search exited successfully, i.e., whether a stepsize
               fulfilling the above Armijo conditions was found
             - x: [np.array] New position after the step
             - f: [float] Objective function value at the new position
             - a: [float] (Sub)optimal stepsize found by the algorithm
             - nfev: [int] Number of objective function calls
    """

    nfev = 0
    x_old = x.copy()

    # Initial stage deciding whether to increase or decrease search step size
    x, _ = projection_callback(x_old + a * d)
    f_new = f(x)
    nfev += 1
    f_target = f_old - alpha * grad.dot(x_old - x)

    if f_new >= f_target:
        # Initial step was too large => decrease a
        for i in range(niter):
            a = a * beta
            x, _ = projection_callback(x_old + a * d)
            f_new = f(x)
            nfev += 1
            f_target = f_old - alpha * grad.dot(x_old - x)
            if f_new < f_target:
                res = {'success': True, 'f': f_new, 'x': x, 'a': a, 'nfev': nfev}
                return SimpleNamespace(**res)
        else:
            res = {'success': False, 'f': f_new, 'x': x, 'a': a, 'nfev': nfev}
            return SimpleNamespace(**res)
    else:
        # Initial step might probably have been larger => try to increase a
        for i in range(niter):
            a_before = a
            x_before = x.copy()
            f_before = f_new
            a = a / beta
            x, _ = projection_callback(x_old + a * d)
            f_new = f(x)
            nfev += 1
            f_target = f_old - alpha * grad.dot(x_old - x)
            if f_new > f_target:
                res = {'success': True, 'f': f_before, 'x': x_before, 'a': a_before, 'nfev': nfev}
                return SimpleNamespace(**res)
        else:
            res = {'success': False, 'f': f_new, 'x': x, 'a': a, 'nfev': nfev}
            return SimpleNamespace(**res)


def adam_spsa(f, x0, projection_callback, c=1e-9, a=0.1, gamma=0.101, alpha=0.602, A_fac=0.05,
              beta_1=0.9, beta_2=0.9, eps=1e-15, niter=1000, verbosity=1):
    """
    Implementation of a Simultaneous Perturbation Stochastic Approximation (SPSA) gradient descent
    algorithm, see
    +++++
    J. C. Spall, An Overview of the Simultaneous Perturbation Method for Efficient Optimization,
    Johns Hopkins APL Technical Digest 19 (1998).
    +++++
    coupled with an Adaptive Moment Estimation (Adam), see
    +++++
    D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, arXiv:1412.6980 (2014).
    +++++
    In addition, here we allow to constrain the search region to a rectangular box. Please note that
    this SPSA implementation was not designed to deal with noisy objective functions, but rather to
    speed up high-dimensional local optimization with expensive cost functions (in n dimensions,
    a standard central differences gradient approximation takes 2n objective function calls, whereas
    the SPSA gradient approximation only takes 2, independent of the problem's dimensionality).

    :param f: [callable] Objective function. Must accept its argument x as numpy array
    :param x0: [np.array] Initial condition
    :param projection_callback: [callable] Bounds projection, see description of parameter
           ``projection_callback`` in :func:`bfgs_b`
    :param c: [float] Initial step size for estimating the gradient approximation
    :param a: [float] Initial "gradient descent" step size
    :param gamma: [float] SPSA gamma determining the decay of the step size for estimating the
           gradient approximation over time. Must be > 0. The larger gamma, the faster the decay
    :param alpha: [float] SPSA gamma determining the decay of the "gradient descent" step size over
           time. Must be > 0. The larger alpha, the faster the decay
    :param A_fac: [float] Offset factor for calculating the SPSA "gradient descent" step size decay.
           Must be > 0. The larger A_fac, the smaller the step size
    :param beta_1: [float] Adam "forgetting factor" for the previous gradient approximations. Must
           be in between 0 and 1
    :param beta_2: [float] Adam "forgetting factor" for the squares of the previous gradient
           approximations. Must be in between 0 and 1
    :param eps: [float] Absolute tolerance
    :param niter: [int] Maximum number of iterations
    :param verbosity: [int] Output verbosity. Must be 0, 1, or 2
    :return: (x_best, f_best, nfev, nit, success, trace), where
             - x_best [np.array] is the best x found so far,
             - f_best [float] is the corresponding objective function value,
             - nfev [int] is the number of objective function evaluations taken,
             - nit [int] is the number of iterations,
             - success [bool] indicates whether the algorithm finished successfully, i.e, whether
               absolute tolerances were met, and
             - trace [np.array] is the optimizer trace, i.e., contains all visited points of x
    """

    assert verbosity in [0, 1, 2], 'verbosity must be 0, 1, or 2.'
    assert np.array_equal(projection_callback(x0)[0], x0), ('x0 outside the bounded domain ' +
                                                            'defined by projection_callback.')

    n_dims = len(x0)
    A = A_fac * niter
    m = v = 0

    trace = np.empty((niter, n_dims))
    f0 = f(x0)
    nfev = 1
    f_best = f0
    x_best = x0.copy()
    x = x0.copy()
    for k in range(niter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma

        # Choose stochastic perturbations for calculating the gradient approximation
        delta = 2 * np.round(np.random.uniform(0, 1, n_dims)) - 1

        # Boundary hit
        _, bounds_hit = projection_callback(x)
        if bounds_hit.any():
            f_minus = f(x - ck * delta)
            f_plus = f(x + ck * delta)
            nfev += 2
            ghat_test = (f_plus - f_minus) / (2 * ck * delta)

            # Check whether following the objective function's gradient would lead to leaving
            # the bounded domain
            bounds_hit_new = projection_callback(x - ak * ghat_test)[1]
            bounds_stuck = np.logical_and(bounds_hit, bounds_hit_new)

            # "Projected" stochastic perturbations vector, with perturbations only parallel to the
            # boundary
            delta = np.where(bounds_stuck, 0, delta)

        # Calculate SPSA gradient approximation
        f_minus = f(x - ck * delta)
        f_plus = f(x + ck * delta)
        nfev += 2
        ghat = (f_plus - f_minus) / (2 * ck * np.where(delta == 0, np.inf, delta))

        # Adam algorithm, with the true gradient replaced by the SPSA gradient approximation
        m = beta_1 * m + (1 - beta_1) * ghat
        v = beta_2 * v + (1 - beta_2) * np.power(ghat, 2)
        m_hat = m / (1 - np.power(beta_1, k + 1))
        v_hat = v / (1 - np.power(beta_2, k + 1))
        x = x - ak * m_hat / (np.sqrt(v_hat) + 1e-9)

        # Clip x to bounded region
        x, _ = projection_callback(x)

        f_new = f(x)
        nfev += 1
        if f_new <= f_best:
            f_best = f_new
            x_best = x.copy()
            a *= 1.5
        else:
            x = x_best.copy()
            a /= 1.5

        trace[k] = x.copy()
        if verbosity == 2:
            print('SPSA step {}:\tx = {}, ghat = {}'.format(k + 1, x, ghat))

        if abs(f_plus - f_minus) < eps:
            nit = k + 1
            success = True
            if verbosity > 0:
                print(f'SPSA Gradient descent target accuracy reached after {nit} steps.')
            break

    else:
        if verbosity > 0:
            print(f'Could not reach desired SPSA gradient descent accuracy after {niter + 1} ' +
                  'iterations. Please try increasing the number of iterations or the tolerance.')
        nit = niter + 1
        success = False
    trace = trace[:nit]

    res = {'x': x_best, 'f': f_best, 'nfev': nfev, 'nit': nit, 'success': success, 'trace': trace}
    return SimpleNamespace(**res)
