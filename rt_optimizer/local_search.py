from types import SimpleNamespace

import numpy as np


def bgfs_b(f, x0, projection_callback, H_start=None, a=1, c=1e-6, niter=100, n_linesearch=20,
           alpha_linesearch=0.5, beta_linesearch=0.5, eps_abs=1e-9, eps_rel=1e-6, verbosity=1):
    """
    ToDo: Write docstring

    :param f:
    :param x0:
    :param projection_callback:
    :param H_start:
    :param a:
    :param c:
    :param niter:
    :param n_linesearch:
    :param alpha_linesearch:
    :param beta_linesearch:
    :param eps_abs:
    :param eps_rel:
    :param verbosity:
    :return:
    """

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

        # Calculate step size
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
                  'try increasing niter_gd or eps_gd.')
        nit = niter + 1
        success = False

    trace = trace[:nit]

    return x_best, f_best, nfev, nit, success, trace


def two_way_linesearch(f, x, grad, d, a, niter, f_old, projection_callback, alpha, beta):
    """
    ToDo: Write docstring

    :param f:
    :param x:
    :param grad:
    :param d:
    :param a:
    :param niter:
    :param f_old:
    :param projection_callback:
    :param alpha:
    :param beta:
    :return:
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


def adam_spsa(f, x0, bound_lower, bound_upper, c=1e-9, a=0.1, gamma=0.101, alpha=0.602, A_fac=0.05,
              beta_1=0.9, beta_2=0.9, eps=1e-15, niter=1000, verbosity=1):
    """
    ToDo: Write docstring

    :param f:
    :param x0:
    :param bound_lower:
    :param bound_upper:
    :param c:
    :param a:
    :param gamma:
    :param alpha:
    :param A_fac:
    :param beta_1:
    :param beta_2:
    :param eps:
    :param niter:
    :param verbosity:
    :return:
    """

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
        delta = 2 * np.round(np.random.uniform(0, 1, n_dims)) - 1

        # Boundary hit
        if (x == bound_lower).any() or (x == bound_upper).any():
            boundary_stuck = np.zeros(n_dims, dtype=bool)
            idcs = np.argwhere((x == bound_lower) | (x == bound_upper))
            for i in idcs:
                delta_i = np.zeros(n_dims)
                delta_i[i] = delta[i]
                f_minus = f(x - ck * delta_i)
                f_plus = f(x + ck * delta_i)
                nfev += 2
                if ((f_plus - f_minus <= 0 and x[i] == bound_upper[i]) or
                        (f_plus - f_minus >= 0 and x[i] == bound_upper[i])):
                    boundary_stuck[i] = True
            delta = np.where(boundary_stuck, 0, delta)

        f_minus = f(x - ck * delta)
        f_plus = f(x + ck * delta)
        nfev += 2
        ghat = (f_plus - f_minus) / (2 * ck * np.where(delta == 0, np.inf, delta))

        # Adam algorithm, with the true gradient replaced by the SPSA
        m = beta_1 * m + (1 - beta_1) * ghat
        v = beta_2 * v + (1 - beta_2) * np.power(ghat, 2)
        m_hat = m / (1 - np.power(beta_1, k + 1))
        v_hat = v / (1 - np.power(beta_2, k + 1))
        x = x - ak * m_hat / (np.sqrt(v_hat) + 1e-9)

        x = np.clip(x, bound_lower, bound_upper)
        f_new = f(x)
        nfev += 1
        if f_new <= f_best:
            f_best = f_new
            x_best = x.copy()
            a *= 10
        else:
            x = x_best.copy()
            a /= 10

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
                  'iterations. Please try increasing niter or the SPSA gradient descent accuracy ' +
                  'threshold.')
        nit = niter + 1
        success = False
    f_final = f(x)
    trace = trace[:nit]
    nfev += 1

    return x, f_final, nfev, nit, success, trace
