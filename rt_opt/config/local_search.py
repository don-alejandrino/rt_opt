"""Configuration for the local search algorithms."""

from dataclasses import dataclass, field


@dataclass
class LineSearchConfig:
    """Configuration for the two-way line search algorithm.

    :param niter: Maximum number of linesearch iterations.
    :param alpha: Line search control parameter alpha. Must be in between 0 and 1.
    :param beta: Line search control parameter beta. Must be in between 0 and 1.
    """

    niter: int = 20
    alpha: float = 0.5
    beta: float = 0.33


@dataclass
class BFGSBConfig:
    """Configuration for the BFGS-B search algorithm.

    :param c: Numerical differentiation step size.
    :param a: Initial search step size.
    :param niter: Maximum number of BFGS iterations.
    :param eps_abs: Absolute tolerance.
    :param eps_rel: Relative tolerance.
    """

    c: float = 1e-6
    a: float | None = None
    niter: int = 100
    eps_abs: float = 1e-9
    eps_rel: float = 1e-6
    linesearch: LineSearchConfig = field(default_factory=lambda: LineSearchConfig())


@dataclass
class AdamSPSAConfig:
    """Configuration for the Adam-SPSA search algorithm.

    :param c: Initial differentiation step size for estimating the gradient
           approximation.
    :param a: Initial "gradient descent" step size.
    :param gamma: SPSA gamma determining the decay of the step size `c` over time.
           Must be > 0. The larger `gamma`, the faster the decay.
    :param alpha: SPSA gamma determining the decay of the "gradient descent" step size
           `a` over time. Must be > 0. The larger `alpha`, the faster the decay.
    :param big_a_fac: Offset factor for calculating the SPSA "gradient descent" step
           size decay. Must be > 0. The larger `big_a_fac`, the smaller the step size.
    :param beta_1: Adam "forgetting factor" for the previous gradient approximations.
           Must be in between 0 and 1.
    :param beta_2: Adam "forgetting factor" for the squares of the previous gradient
           approximations. Must be in between 0 and 1.
    :param eps: Absolute tolerance for the gradient magnitude. Once the gradient
           approximation becomes smaller than `eps` for `n_repeated_eps_threshold_hits`
           in a row, the algorithm stops.
    :param n_repeated_eps_threshold_hits: Number of times the gradient approximation
           must be below `eps` in a row before stopping. We do this to avoid stopping
           too early due to stochastic fluctuations in the gradient approximation.
    :param niter: Maximum number of iterations.
    :param seed: Random seed for reproducibility.
    """

    c: float = 1e-9
    a: float = 0.1
    gamma: float = 0.101
    alpha: float = 0.602
    big_a_fac: float = 0.05
    beta_1: float = 0.9
    beta_2: float = 0.9
    eps: float = 1e-15
    n_repeated_eps_threshold_hits: int = 10
    niter: int = 1000
    seed: int | None = None
