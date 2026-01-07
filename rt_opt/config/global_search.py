"""Configuration for the global search algorithms."""

from dataclasses import dataclass


@dataclass
class RunAndTumbleConfig:
    """Configuration for the run-and-tumble search algorithm.

    :param niter: Maximum number of run-and-tumble steps.
    :param stepsize_start: Defines the initial length of a "run" step. If not provided,
           the algorithm tries to auto-scale this length to the problem's scale.
    :param stepsize_decay_fac: Factor by which the run-and-tumble stepsize has decayed
           in the last run-and-tumble iteration compared to its initial value. The
           actual stepsize decreases quadratically from `stepsize_start` to
           `stepsize_start` * `stepsize_decay_fac`.
    :param base_tumble_rate: "Undisturbed" tumble rate when a bacterium does not feel
           any change in attractant concentration.
    :param stationarity_window: If the mean position of all bacteria has had a relative
           change less than `eps_stat` over a step window `stationarity_window`, the
           bacteria distribution is considered to be stationary and the algorithm stops.
    :param eps_stat: See description of parameter `stationarity_window`.
    :param stationarity_r_value_threshold: Threshold for the r-value (the square of
          which equals the Pearson coefficient of determination) in the linear
          regression used for stationarity detection. When the r-value is smaller than
          this threshold, we conclude that there is not yet any (locally) linear trend
          in the mean bacteria position over the considered stationarity window. In this
          case, we assume that a stationary state has not been reached yet.
    :param attraction: Whether the bacteria attract each other or not. We model bacteria
           attraction the following way: Each bacterium is supposed to leave some kind
           of magic attractant at the places it has visited thus far, that attracts all
           other bacteria.
    :param attraction_window: Defines the number of recent positions in a bacterium's
           trace that contributes to the attraction mechanism. We have to define this
           cut-off  length, since otherwise calculating the bacteria attractions becomes
           computationally very expensive. This parameter only has an effect if
           `attraction == True`.
    :param attraction_sigma: The bacterial attractant concentration is modeled to decay
           according to a Gaussian distribution,
           ```
           attraction_strength / 2 / attraction_sigma ** 2
                * exp(-(np.square(x - x_vis) / 2 / attraction_sigma ** 2)),
           ```
           around each point `x_vis` visited thus far. This parameter only has an effect
           if `attraction == True`. Note also that if `attraction_sigma` is not
           provided, the algorithm tries to auto-scale this length to the problem's
           scale.
    :param attraction_strength: See description of parameter `attraction_sigma`. Note
           that if `attraction_strength < 0`, the bacterial attraction turns into a
           repulsion. This parameter only has an effect if `attraction == True`.
    :param bounds_reflection: Whether bacteria reverse their direction when hitting a
           boundary (True) or tumble randomly (False).
    :param seed: Random seed for reproducibility.
    """

    niter: int = 400
    stepsize_start: float | None = None
    stepsize_decay_fac: float = 1e-3
    base_tumble_rate: float = 0.1
    stationarity_window: int = 20
    eps_stat: float = 1e-3
    stationarity_r_value_threshold: float = 0.9
    attraction: bool = False
    attraction_window: int = 10
    attraction_sigma: float | None = None
    attraction_strength: float = 0.5
    bounds_reflection: bool = False
    seed: int | None = None
