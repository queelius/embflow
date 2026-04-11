"""embflow: A calculus over embedding sequences.

Operations on ordered sequences of embedding vectors: projection (fold),
trajectory (scan), derivatives (velocity, curvature), segmentation,
and comparison. Lenses are composable weight functions that control
how sequences aggregate.

Core types:
    Item     = (vector, metadata)
    Seq      = ordered sequence of Items
    Lens     = weight function family, composable via *

Core operations:
    lens.project(vectors, meta)     -> vector (fold)
    lens.trajectory(vectors, meta)  -> vectors (scan)
    velocity(trajectory)            -> vectors (1st derivative)
    curvature(trajectory)           -> vectors (2nd derivative)
    segment(vectors, boundaries)    -> list of vector arrays
    drift(trajectory)               -> float
    adaptive_alpha(vectors)         -> float
"""
__version__ = "0.1.0"

from embflow.lens import (
    Lens,
    Uniform,
    Exponential,
    ReverseExponential,
    Gaussian,
    Surprise,
    FieldWeight,
    TimeDecay,
    Custom,
)
from embflow.ops import (
    velocity,
    curvature,
    speed,
    angular_velocity,
    drift,
    peaks,
    segment,
    auto_segment,
    adaptive_alpha,
    structural_richness,
)
from embflow.compare import (
    trajectory_distance,
    continuation_score,
)
