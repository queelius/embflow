"""embflow: a calculus over embedding sequences.

Treats an ordered sequence of embedding vectors as a path in R^d.
Provides weighted folds, scan-style smoothing, differential operators
(velocity, curvature, jerk, angular velocity, speed), second-order
geometry (arc length, local curvature radius, velocity covariance),
segmentation, and trajectory distance.

Primary object: an (n, d) ndarray representing a sequence of vectors.

Core operations:
    weighted_mean(vectors, weights)         -> vector (fold)
    smooth_exponential(vectors, alpha)      -> vectors (O(n) scan)
    velocity / curvature / jerk             -> differential operators
    arc_length / speed / angular_velocity   -> scalar path metrics
    local_curvature_radius                  -> osculating-circle radius
    velocity_covariance                     -> local structure tensor
    trajectory_distance(a, b, method=...)   -> distance between two paths
    continuation_score(a, b)                -> does B pick up where A left off?
"""
__version__ = "0.2.0"

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
