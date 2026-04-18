"""Random Sequential Embeddings implementation."""

import logging
import warnings
from functools import partial

import numpy as np

from rt_opt.config.optimizer import SequentialRandomEmbeddingsConfig
from rt_opt.protocols.optimizer import OptimizerProtocol
from rt_opt.search.search_output import SingleSearchOutput
from rt_opt.utils.types import ObjectiveFunctionType, ProjectionCallbackType


def sequential_random_embeddings(
    f: ObjectiveFunctionType,
    x0_population: np.ndarray,
    projection_callback: ProjectionCallbackType,
    optimizer: OptimizerProtocol,
    config: SequentialRandomEmbeddingsConfig | None = None,
) -> SingleSearchOutput:
    """Random Sequential Embeddings implementation.

    Implementation of the Sequential Random Embeddings algorithm described in
    +++++
    H. Qian, Y.-Q. Hu, and Y. Yu, Derivative-Free Optimization of High-Dimensional
    Non-Convex Functions by Sequential Random Embeddings, Proceedings of the
    Twenty-Fifth International Joint Conference on Artificial Intelligence, AAAI Press
    (2016).
    +++++
    The idea is basically to reduce high-dimensional problems to low-dimensional ones by
    embedding the original, high-dimensional search space ℝ^h into a low dimensional
    one, ℝ^l, by sequentially applying the random linear transformation
    ```
    x(n+1) = α(n+1)x(n) + A•y(n+1),    x ∈ ℝ^h, y ∈ ℝ^l, A ∈ N(0, 1)^(h×l), α ∈ ℝ,
    ```
    and minimizing the objective function f(αx + A•y) w.r.t. (α, y).

    :param f: Objective function. Must accept its argument `x` as numpy array.
    :param x0_population: Initial values for the bacteria population in the original,
           high-dimensional space ℝ^h. Must have the shape (n_bacteria, h).
    :param projection_callback: Bounds projection, see description of parameter
           `projection_callback` in :func:`search.local_search.bfgs_b`.
    :param optimizer: Optimizer function to be used for minimizing the target function
           in the embedded space.
    :param config: Configuration for the Sequential Random Embeddings algorithm.
    :return: A `SingleSearchOutput` instance containing the search results in the
             embedded space.

    """
    logger = logging.getLogger(__name__)
    if config is None:
        config = SequentialRandomEmbeddingsConfig()

    rng = np.random.default_rng(config.seed)
    n_embeddings = config.n_embeddings

    # The effective dimension of the embedded problem is ℝ^(l+1), since we optimize
    # f(αx + A•y) w.r.t. the tuple (α, y).
    n_reduced_dims_eff = config.n_reduced_dims + 1

    orig_dim = x0_population.shape[1]
    x = np.zeros(orig_dim)
    x_best = x.copy()
    f_best = np.inf
    nfev = nit = 0
    success_best = False
    for i in range(n_embeddings):
        a_matrix = rng.normal(size=(orig_dim, n_reduced_dims_eff - 1))

        # Normalize rows of `a_matrix`
        normalization_sum = a_matrix.sum(axis=1)
        normalization_sum = np.where(normalization_sum == 0, 1, normalization_sum)
        a_matrix = a_matrix / normalization_sum[:, np.newaxis]

        def f_embedded(
            x_embedded: np.ndarray, x_: np.ndarray, a_matrix_: np.ndarray
        ) -> float:
            return f(
                projection_callback(x_embedded[0] * x_ + a_matrix_.dot(x_embedded[1:]))[
                    0
                ]
            )

        # Set up bounds callback
        def bounds_embedded(
            x_embedded: np.ndarray, x_: np.ndarray, a_matrix_: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            bounds_hit = np.zeros(len(x_embedded), dtype=bool)
            x_proj, bounds_hit_orig = projection_callback(
                x_embedded[0] * x_ + a_matrix_.dot(x_embedded[1:])
            )
            if bounds_hit_orig.any():  # Boundary hit in original, non-embedded variable
                x_embedded[1:] = np.linalg.lstsq(
                    a_matrix_, x_proj - x_embedded[0] * x_, rcond=None
                )[0]
                bounds_hit[1:] = (a_matrix_[bounds_hit_orig] != 0).any(axis=0)

            return x_embedded, bounds_hit

        def bounds_embedded_population(
            x_embedded: np.ndarray, x_: np.ndarray, a_matrix_: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            out = np.array(
                [bounds_embedded(x_single, x_, a_matrix_) for x_single in x_embedded]
            )
            return out[:, 0], out[:, 1]

        # Set up y0
        y0 = np.zeros((x0_population.shape[0], n_reduced_dims_eff))
        y0[:, 0] = 1
        y0[:, 1:] = np.array(
            [
                np.linalg.lstsq(a_matrix, x_orig - x, rcond=None)[0]
                for x_orig in x0_population
            ]
        )

        info_msg = f"\nEmbedding iteration {i}"
        logger.info(info_msg)
        logger.info("-" * len(info_msg))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found initial conditions outside the defined search domain.",
            )
            res_embedded = optimizer.run(
                f=partial(f_embedded, x_=x, a_matrix_=a_matrix),
                x0_population=y0,
                projection_callback=partial(bounds_embedded, x_=x, a_matrix_=a_matrix),
                projection_callback_population=partial(
                    bounds_embedded_population, x_=x, a_matrix_=a_matrix
                ),
            )
        y = res_embedded.x_best
        f_val = res_embedded.f_best
        nfev += res_embedded.nfev
        nit += res_embedded.nit

        x = projection_callback(y[0] * x + a_matrix.dot(y[1:]))[0]

        logger.info("Random embedding gave x = %s.", str(x))

        if f_val < f_best:
            f_best = f_val
            x_best = x.copy()
            success_best = res_embedded.success

    return SingleSearchOutput(
        x_best=x_best,
        f_best=f_best,
        nfev=nfev,
        nit=nit,
        success=success_best,
        trace=None,
    )
