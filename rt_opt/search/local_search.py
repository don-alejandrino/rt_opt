"""Algorithms for local, gradient-based minimum search."""

import logging
from warnings import warn

import numpy as np

from rt_opt.config.local_search import AdamSPSAConfig, BFGSBConfig, LineSearchConfig
from rt_opt.search.search_output import LineSearchOutput, SingleSearchOutput
from rt_opt.utils.types import ObjectiveFunctionType, ProjectionCallbackType

logger = logging.getLogger(__name__)


def bfgs_b(  # noqa: PLR0915, C901
    f: ObjectiveFunctionType,
    x0: np.ndarray,
    projection_callback: ProjectionCallbackType,
    hessian_start: np.ndarray | None = None,
    config: BFGSBConfig | None = None,
) -> SingleSearchOutput:
    """BFGS-B algorithm for bounded local optimization.

    This BFGS-B implementation can deal with arbitrarily bounded search regions. An
    estimate of the optimal step size for each iteration is found using a
    two-way-backtracking line search algorithm.

    :param f: Objective function. Must accept its argument x as numpy array.
    :param x0: Initial condition.
    :param projection_callback: Bounds projection. The function `projection_callback(x)`
           must return a tuple `(x_projected, bounds_hit)`, where `x_projected` is the
           input variable `x` projected to the defined search region. That is, if `x` is
           within this region, it is returned unchanged, whereas if it is outside this
           region, it is projected to the region's boundaries. The second output,
           `bounds_hit`, indicates whether the boundary has been hit for each component
           of `x`. If, for example, `x` is three-dimensional and has hit the search
           region's boundaries in `x[1]` and `x[2]`, but not in `x[3]`,
           `bounds_hit = [True, True, False]`. Note that the search domain needs not
           necessarily be rectangular. Therefore, we define a "boundary hit" in any
           component of `x` in the following way: `bounds_hit[i] = True` iff either
           `x` + δê_i or `x` - δê_i is outside the defined search domain ∀ δ ∈ ℝ⁺, where
           ê_i is the i-th unit vector.
    :param hessian_start: Initial Hessian at `x0`. If known, can be used to "warm-start"
           the algorithm for faster convergence. Otherwise (i.e., if set to None), the
           identity matrix is used.
    :param config: Configuration provided to the BFGS-B algorithm.
    :return: A `SingleSearchOutput` instance containing the search results.

    """
    if config is None:
        config = BFGSBConfig()

    if config.a is None:
        warn(
            " Initial search step size `a` was not provided. As auto-scaling "
            "of `a` is not implemented for the low-level search routines, it has been "
            "set to 1 by default. Please consider tuning this parameter to your "
            "specific problem.",
            stacklevel=1,
        )
        a = 1
    else:
        a = config.a

    def calculate_gradient(x_: np.ndarray, delta: float) -> np.ndarray:
        gradient = np.zeros(n_dims)
        for m in range(n_dims):
            unit_vec = np.zeros(n_dims)
            unit_vec[m] = 1
            f_minus = f(x_ - delta * unit_vec)
            f_plus = f(x_ + delta * unit_vec)
            gradient[m] = (f_plus - f_minus) / 2 / delta

        return gradient

    n_dims = len(x0)
    trace = np.empty((config.niter, n_dims))

    b = _initialize_inverse_hessian(hessian_start, n_dims)

    nfev = 0
    x = x0.copy()
    x, _ = projection_callback(x)
    x_best = x.copy()
    grad = calculate_gradient(x, config.c)
    nfev += 2 * n_dims
    f_curr = f_best = f(x)
    nfev += 1
    acc0 = np.linalg.norm(x - projection_callback(x - grad)[0])
    if not isinstance(acc0, float):
        err_msg = f"`x0` must be a vector of shape (n_dims,), but got {x0.shape}."
        raise ValueError(err_msg)  # noqa: TRY004

    for k in range(config.niter):
        _, bounds_hit = projection_callback(x)
        b[bounds_hit] = np.identity(n_dims)[bounds_hit]

        # Calculate search direction
        d = -b.dot(grad)

        # Calculate optimal step size and update x
        a_old = a
        ls_result = two_way_linesearch(
            f,
            x,
            grad,
            d,
            a,
            f_curr,
            projection_callback,
            config.linesearch,
        )
        f_curr = ls_result.f
        x = ls_result.x
        a = ls_result.a
        nfev += ls_result.nfev

        if not ls_result.success:
            logger.warning(
                "BGFS step %d: Couldn't find sufficiently good step size during "
                "%d line search steps.",
                k + 1,
                config.linesearch.niter,
            )

        if f_curr < f_best:
            f_best = f_curr
            x_best = x.copy()
        else:
            a = a_old * 0.95

        # Update inverse Hessian approximation
        s = a * d
        s[bounds_hit] = 0
        grad_new = calculate_gradient(x, config.c)
        nfev += 2 * n_dims
        y = grad_new - grad
        y[bounds_hit] = 0
        if np.dot(y, s) <= 0:
            b = np.identity(n_dims)
        else:
            b = (np.identity(n_dims) - np.outer(s, y) / np.dot(y, s)).dot(b).dot(
                np.identity(n_dims) - np.outer(y, s) / np.dot(y, s)
            ) + np.outer(s, s) / np.dot(y, s)

        grad = grad_new
        trace[k] = x.copy()
        logger.debug("BGFS step %d:\tx = %s, f(x) = %g", k + 1, str(x), f_curr)

        acc = np.linalg.norm(x - projection_callback(x - grad)[0])
        if acc <= config.eps_abs + config.eps_rel * acc0:
            nit = k + 1
            success = True
            logger.info("BGFS target accuracy reached after %d steps.", nit)
            break

    else:
        logger.warning(
            "Could not reach desired BGFS accuracy after %d iterations. Please "
            "try increasing the number of iterations or the tolerance.",
            config.niter + 1,
        )
        nit = config.niter + 1
        success = False

    trace = trace[:nit]

    return SingleSearchOutput(x_best, f_best, nfev, nit, success, trace)


def _initialize_inverse_hessian(hessian: np.ndarray | None, n_dims: int) -> np.ndarray:
    # If the elements of the provided Hessian are all close to zero, inverting it
    # becomes numerically unstable. In this case, we fall back to the identity matrix.
    hessian_nonzero_threshold = 1e-3
    if hessian is None or np.max(np.abs(hessian)) < hessian_nonzero_threshold:
        b = np.identity(n_dims)
    else:
        if hessian.shape != (n_dims, n_dims):
            err_msg = (
                "Provided Hessian has wrong format. Expected shape "
                f"({n_dims}, {n_dims}), but got {hessian.shape}."
            )
            raise ValueError(err_msg)
        try:
            b = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            b = np.identity(n_dims)

    return b


def two_way_linesearch(  # noqa: PLR0913
    f: ObjectiveFunctionType,
    x: np.ndarray,
    grad: np.ndarray,
    d: np.ndarray,
    a: float,
    f_old: float,
    projection_callback: ProjectionCallbackType,
    config: LineSearchConfig | None = None,
) -> LineSearchOutput:
    """Two-way backtracking line search.

    Implementation of a two-way-backtracking line search algorithm, as outlined in
    +++++
    T. T. Truong, T. H. Nguyen, Backtracking gradient descent method for general C1
    functions, with applications to Deep Learning, arXiv:1808.05160 (2018).
    +++++
    Here, we also include a projection of x onto a bounded subspace. The Armijo
    condition deciding whether a stepsize a is accepted reads in this case:
    f(P(x + a * d)) ≤ f(x) - alpha * ∇f(x)•(x - P(x + a * d)),
    where d is the search direction and P the projection of x onto a bounded subspace.

    :param f: Objective function. Must accept its argument x as numpy array.
    :param x: Current (starting) position.
    :param grad: Gradient of f at the current (starting) position x.
    :param d: Search direction.
    :param a: Initial search step size.
    :param f_old: Objective function value at the beginning of the line search, f(x).
           We pass it as an argument to avoid redundant function calls.
    :param projection_callback: [callable] Bounds projection, see description of
           parameter `projection_callback` in :func:`bfgs_b`
    :param config: Configuration provided to the line search algorithm.
    :return: A `LineSearchOutput` instance containing the search results.

    """
    if config is None:
        config = LineSearchConfig()

    nfev = 0
    x_old = x.copy()

    # Initial stage deciding whether to increase or decrease search step size
    x, _ = projection_callback(x_old + a * d)
    f_new = f(x)
    nfev += 1
    f_target = f_old - config.alpha * grad.dot(x_old - x)

    if f_new >= f_target:
        # Initial step was too large => decrease a
        for i in range(config.niter):
            a *= config.beta
            x, _ = projection_callback(x_old + a * d)
            f_new = f(x)
            nfev += 1
            f_target = f_old - config.alpha * grad.dot(x_old - x)
            if f_new < f_target:
                return LineSearchOutput(
                    x=x, f=f_new, a=a, nfev=nfev, nit=i + 1, success=True
                )
        return LineSearchOutput(
            x=x, f=f_new, a=a, nfev=nfev, nit=config.niter, success=False
        )

    # Initial step might probably have been larger => try to increase a
    for i in range(config.niter):
        a_before = a
        x_before = x.copy()
        f_before = f_new
        a /= config.beta
        x, _ = projection_callback(x_old + a * d)
        f_new = f(x)
        nfev += 1
        f_target = f_old - config.alpha * grad.dot(x_old - x)
        if f_new > f_target:
            return LineSearchOutput(
                x=x_before, f=f_before, a=a_before, nfev=nfev, nit=i + 1, success=True
            )

    return LineSearchOutput(
        x=x, f=f_new, a=a, nfev=nfev, nit=config.niter, success=False
    )


def adam_spsa(  # noqa: PLR0915
    f: ObjectiveFunctionType,
    x0: np.ndarray,
    projection_callback: ProjectionCallbackType,
    config: AdamSPSAConfig | None = None,
) -> SingleSearchOutput:
    """SPSA gradient descent with Adam optimizer.

    Implementation of a Simultaneous Perturbation Stochastic Approximation (SPSA)
    gradient descent algorithm, see
    +++++
    J. C. Spall, An Overview of the Simultaneous Perturbation Method for Efficient
    Optimization, Johns Hopkins APL Technical Digest 19 (1998),
    +++++
    coupled with an Adaptive Moment Estimation (Adam), see
    +++++
    D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization,
    arXiv:1412.6980 (2014).
    +++++
    In addition, here we allow to constrain the search region to a rectangular box.
    Please note that this SPSA implementation was not designed to deal with noisy
    objective functions, but rather to speed up high-dimensional local optimization with
    expensive cost functions (in n dimensions, a standard central differences gradient
    approximation takes 2n objective function calls, whereas the SPSA gradient
    approximation only takes 2, independent of the problem's dimensionality).

    :param f: Objective function. Must accept its argument x as numpy array.
    :param x0: Initial condition.
    :param projection_callback: Bounds projection, see description of parameter
           `projection_callback` in :func:`bfgs_b`.
    :param config: Configuration provided to the Adam-SPSA algorithm.
    :return: A `SingleSearchOutput` instance containing the search results.

    """
    if config is None:
        config = AdamSPSAConfig()

    if not np.array_equal(projection_callback(x0)[0], x0):
        err_msg = "`x0` is outside the bounded domain defined by `projection_callback`."
        raise ValueError(err_msg)

    rng = np.random.default_rng(config.seed)

    n_dims = len(x0)
    big_a = config.big_a_fac * config.niter
    a = config.a
    m = v = 0

    trace = np.empty((config.niter, n_dims))
    f0 = f(x0)
    nfev = 1
    f_best = f0
    x_best = x0.copy()
    x = x0.copy()
    for k in range(config.niter):
        ak = a / (k + 1 + big_a) ** config.alpha
        ck = config.c / (k + 1) ** config.gamma

        # Choose stochastic perturbations for calculating the gradient approximation
        delta = 2 * np.round(rng.uniform(0, 1, n_dims)) - 1

        # Boundary hit
        _, bounds_hit = projection_callback(x)
        if bounds_hit.any():
            f_minus = f(x - ck * delta)
            f_plus = f(x + ck * delta)
            nfev += 2
            ghat_test = (f_plus - f_minus) / (2 * ck * delta)

            # Check whether following the objective function's gradient would lead to
            # leaving the bounded domain
            bounds_hit_new = projection_callback(x - ak * ghat_test)[1]
            bounds_stuck = np.logical_and(bounds_hit, bounds_hit_new)

            # "Projected" stochastic perturbations vector, with perturbations only
            # parallel to the boundary
            delta = np.where(bounds_stuck, 0, delta)

        # Calculate SPSA gradient approximation
        f_minus = f(x - ck * delta)
        f_plus = f(x + ck * delta)
        nfev += 2
        ghat = (f_plus - f_minus) / (2 * ck * np.where(delta == 0, np.inf, delta))

        # Adam algorithm, with the true gradient replaced by the SPSA gradient
        # approximation
        m = config.beta_1 * m + (1 - config.beta_1) * ghat
        v = config.beta_2 * v + (1 - config.beta_2) * np.power(ghat, 2)
        m_hat = m / (1 - np.power(config.beta_1, k + 1))
        v_hat = v / (1 - np.power(config.beta_2, k + 1))
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
        logger.debug("SPSA step %d:\tx = %s, ghat = %s", k + 1, str(x), str(ghat))

        if abs(f_plus - f_minus) < config.eps:
            nit = k + 1
            success = True
            logger.info(
                "SPSA Gradient descent target accuracy reached after %d steps.", nit
            )
            break

    else:
        logger.info(
            "Could not reach desired SPSA gradient descent accuracy after %d "
            "iterations. Please try increasing the number of iterations or the "
            "tolerance.",
            config.niter + 1,
        )
        nit = config.niter + 1
        success = False
    trace = trace[:nit]

    return SingleSearchOutput(x_best, f_best, nfev, nit, success, trace)
