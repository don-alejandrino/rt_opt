"""Configuration for the optimizer."""

from dataclasses import dataclass, field
from typing import Literal

from rt_opt.config.global_search import RunAndTumbleConfig
from rt_opt.config.local_search import BFGSBConfig


@dataclass
class SequentialRandomEmbeddingsConfig:
    """Configuration for the Sequential Random Embeddings algorithm.

    :param n_embeddings: Number of embedding iterations when using Sequential Random
           Embeddings.
    :param n_reduced_dims: Dimension of the embedded problem, i.e., target dimension
           after applying Sequential Random Embeddings.
    :param seed: Random seed for reproducibility.
    """

    n_embeddings: int = 5
    n_reduced_dims: int = 2
    seed: int | None = None


@dataclass
class OptimizationConfig:
    """Configuration for the high-level optimization algorithm.

    :param init: Determines how initial bacteria positions are sampled from Ω if
           `x0` is None, see description of parameter `x0`. Currently supported:
           "random" and "uniform".
    :param n_bacteria_per_dim: How many bacteria to spawn in each dimension. Note that
           the total number of bacteria is
           i)  n_bacteria = n_bacteria_per_dim ** n_dims if n_dims <= max_dims or
           ii) n_bacteria = n_bacteria_per_dim ** (n_reduced_dims + 1)
               if n_dims > max_dims.
           If `x0` is provided with shape (n_bacteria, n_dims), n_bacteria should agree
           with this relation.
    :param n_best_selection: At the end of the run-and-tumble exploration stage, a local
           gradient-based search is performed, starting from the best positions found
           thus far by the `n_best_selection` best bacteria.
    :param max_dims: Maximum dimension of problems to be solved without using Sequential
           Random Embeddings.
    :param global_search: Configuration for the global search stage.
    :param local_search: Configuration for the local search stage.
    :param embedding: Configuration for the Sequential Random Embeddings stage. Only has
           an effect if n_dims > `max_dims`.
    :param seed: Random seed for reproducibility.
    """

    init: Literal["random", "uniform"] = "uniform"
    n_bacteria_per_dim: int = 3
    n_best_selection: int = 3
    max_dims: int = 3
    global_search: RunAndTumbleConfig = field(
        default_factory=lambda: RunAndTumbleConfig()
    )
    local_search: BFGSBConfig = field(default_factory=lambda: BFGSBConfig())
    embedding: SequentialRandomEmbeddingsConfig = field(
        default_factory=lambda: SequentialRandomEmbeddingsConfig()
    )
    seed: int | None = None
