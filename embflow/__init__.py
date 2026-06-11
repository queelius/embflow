"""embflow: a calculus over embedding sequences.

Treats an ordered sequence of embedding vectors as a path in R^d.

The core dynamics are a leaky integrator: s_k = alpha*s_{k-1} + m_k*e_k
(the Euler step of dx/dt = -lambda*x + f(t), alpha = e^(-lambda dt));
unit normalization x_k = s_k/||s_k|| is a readout for cosine comparison,
not part of the dynamics. Lens convention everywhere:
w(j,k) = alpha^(k-j) — HIGHER alpha = LONGER memory, alpha -> 1 is the
running mean, half-life log(0.5)/log(alpha) steps.

Primary object: an (n, d) ndarray representing a sequence of vectors.

Core operations:
    leaky_state / trajectory                -> linear state / normalized readout
    weighted_mean(vectors, weights)         -> vector (fold)
    smooth_exponential(vectors, alpha)      -> vectors (O(n) scan)
    velocity / curvature / jerk             -> differential operators
    speed / angular_velocity / turning_cosines -> scalar motion signals
    tortuosity / speed_autocorr / motion_signature -> per-sequence gait
    arc_length / drift / local_curvature_radius / velocity_covariance
    adaptive_alpha                          -> fitted memory length
    shuffle / role_slot_shuffle / null_corrected -> order-vs-composition nulls
    trajectory_distance / velocity_gram     -> trajectory comparison
    continuation_score(a, b)                -> does B pick up where A left off?
    prefix_experiment(convs, embed_fn)      -> the prefix-validation protocol
    openai_embed_fn / ollama_embed_fn / cached_embed_fn -> embedding backends
"""
__version__ = "0.3.0"

from embflow.weights import (
    uniform_weights,
    exponential_weights,
    reverse_exponential_weights,
    gaussian_weights,
    time_decay_weights,
    field_weights,
    novelty_weights,
    weighted_mean,
)
from embflow.smooth import (
    smooth_uniform,
    smooth_exponential,
    smooth_reverse_exponential,
    smooth_gaussian,
    smooth_time_decay,
)
from embflow.state import (
    leaky_state,
    trajectory,
    alpha_to_half_life,
    half_life_to_alpha,
)
from embflow.ops import (
    ALPHA_GRID,
    velocity,
    curvature,
    jerk,
    speed,
    angular_velocity,
    turning_cosines,
    tortuosity,
    speed_autocorr,
    arc_length,
    drift,
    local_curvature_radius,
    velocity_covariance,
    peaks,
    segment,
    auto_segment,
    adaptive_alpha,
    motion_signature,
    structural_richness,
)
from embflow.nulls import (
    shuffle,
    role_slot_shuffle,
    null_corrected,
    paired_stats,
)
from embflow.compare import (
    trajectory_distance,
    velocity_gram,
    continuation_score,
)
from embflow.backends import (
    EmbedFn,
    cached_embed_fn,
    openai_embed_fn,
    ollama_embed_fn,
)
from embflow.validate import (
    lens_weights,
    default_lenses,
    prefix_experiment,
)
