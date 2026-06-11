"""Differential and geometric operators for trajectories in embedding space.

Input is an (n, d) ndarray: a sequence of d-dimensional vectors treated
as a path in R^d. Inputs can be raw embeddings or smoothed trajectories
from ``embflow.smooth``. Operators cover first/second/third derivatives,
arc-length and turning rate, second-order structure (local curvature
radius and velocity covariance), segmentation, and adaptive analysis.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

from embflow.smooth import smooth_exponential
from embflow.weights import (
    exponential_weights,
    novelty_weights,
    reverse_exponential_weights,
    uniform_weights,
    weighted_mean,
)


EPS = 1e-8

# Adaptive-alpha grid from the embedding-dynamics paper's motionlib:
# 0.05..0.95 in steps of 0.05, plus 0.999 (near-running-mean).
ALPHA_GRID = np.concatenate([np.arange(0.05, 1.00, 0.05), [0.999]])
ALPHA_MAX_MESSAGES = 400


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos_sim_scalar(a, b):
    """Scalar cosine similarity between two 1D vectors."""
    return float(_cosine_sim(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def _auto_threshold(signal, k=1.5):
    """Automatic threshold: mean + k * std of the signal.

    The default multiplier (k=1.5) matches scipy-style peak pickers.
    Callers that want a less strict default pass a smaller k.
    """
    return float(np.mean(signal) + k * np.std(signal))


# === Differential operators ===

def velocity(trajectory):
    """First difference: displacement between consecutive points.

    v_k = trajectory[k+1] - trajectory[k]   for k = 0..n-2

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-1, d)
    """
    return np.diff(trajectory, axis=0)


def curvature(trajectory):
    """Second difference: rate of change of direction.

    c_k = trajectory[k+2] - 2*trajectory[k+1] + trajectory[k]

    High-magnitude curvature = the trajectory is turning sharply.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-2, d)
    """
    return np.diff(trajectory, n=2, axis=0)


def jerk(trajectory):
    """Third difference: rate of change of curvature.

    Sensitive to the sharpest local changes, more so than curvature
    because it measures how suddenly the bending itself changes.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-3, d)
    """
    return np.diff(trajectory, n=3, axis=0)


def speed(trajectory):
    """Scalar speed at each step: ||velocity||.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-1,)
    """
    return np.linalg.norm(velocity(trajectory), axis=1)


def angular_velocity(trajectory):
    """Angular change between consecutive trajectory points.

    a_k = 1 - cos(trajectory[k], trajectory[k+1])

    For a unit-normalized trajectory, this is a measure of how much
    the position on the unit sphere rotates between steps.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-1,)
    """
    n = len(trajectory)
    angles = np.empty(n - 1)
    for k in range(n - 1):
        angles[k] = 1 - _cos_sim_scalar(trajectory[k], trajectory[k + 1])
    return angles


def turning_cosines(trajectory):
    """Cosine between consecutive VELOCITY vectors: cos(v_k, v_{k+1}).

    +1 = continuing straight, 0 = right-angle turn, -1 = full reversal.
    NaN where either velocity is ~0 (consecutive duplicate points).

    NOT the same as ``angular_velocity``, which is 1 - cos between
    consecutive trajectory POINTS (how far the position rotates);
    this measures how much the direction of MOTION turns.

    Null fact (see ``embflow.nulls``): for exchangeable unit vectors the
    expected turning cosine is exactly -1/2, independent of anisotropy.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-2,), NaN at degenerate steps.
    """
    v = np.diff(np.asarray(trajectory, dtype=float), axis=0)
    norms = np.linalg.norm(v, axis=1)
    num = (v[:-1] * v[1:]).sum(axis=1)
    den = norms[:-1] * norms[1:]
    out = np.full(len(num), np.nan)
    ok = den > EPS
    out[ok] = num[ok] / den[ok]
    return out


def tortuosity(trajectory, window=8):
    """Mean over sliding windows of (net displacement / path length).

    1.0 = perfectly directed over each window; near 0 = the path mostly
    backtracks. Ported from the embedding-dynamics motionlib (validated
    2026-06-10: real conversations are more directed than shuffled,
    paired Cohen's d = +1.39).

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)
    window : int
        Window length in steps; clamped to n-1.

    Returns
    -------
    float, NaN when fewer than 2 steps or all windows are stationary.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    n = len(trajectory)
    if n - 1 < 2:
        return float("nan")
    w = min(window, n - 1)
    s = speed(trajectory)
    nets = np.linalg.norm(trajectory[w:] - trajectory[:-w], axis=1)
    csum = np.concatenate([[0.0], np.cumsum(s)])
    paths = csum[w:] - csum[:-w]
    ok = paths > EPS
    if not ok.any():
        return float("nan")
    return float((nets[ok] / paths[ok]).mean())


def _lag_autocorr(signal, lag=1):
    """Lag-k autocorrelation of a 1D signal; NaN entries dropped first.

    NaN when fewer than lag+2 finite values or the signal is constant.
    """
    x = signal[~np.isnan(signal)]
    if len(x) < lag + 2 or x.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def speed_autocorr(trajectory, lag=1):
    """Lag-k autocorrelation of the speed series ||v_k||.

    Measures burstiness of motion: do large steps cluster? CAUTION
    (validated 2026-06-10): this is NOT a pure order statistic — message
    eccentricity makes both adjacent steps large regardless of order, so
    even the exchangeable null has positive speed autocorrelation.
    Report it null-corrected (see ``embflow.nulls.null_corrected``).

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)
    lag : int

    Returns
    -------
    float, NaN on degenerate input.
    """
    return _lag_autocorr(speed(np.asarray(trajectory, dtype=float)), lag)


# === Global and second-order geometric measures ===

def arc_length(trajectory):
    """Cumulative path length along the trajectory.

    Returns a (n,) array where result[k] is the total Euclidean distance
    traveled from trajectory[0] to trajectory[k]. result[0] = 0.

    Euclidean distance is used throughout; for unit-normalized
    trajectories, segment distances relate to angles on the unit
    sphere via |u - v| = 2 sin(theta/2).

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n,)
    """
    if len(trajectory) < 2:
        return np.zeros(len(trajectory))
    segments = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segments)])


def drift(trajectory):
    """Cosine distance from the first to the last trajectory point.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    float
    """
    if len(trajectory) < 2:
        return 0.0
    return 1 - _cos_sim_scalar(trajectory[0], trajectory[-1])


def local_curvature_radius(trajectory, eps=1e-12):
    """Radius of the osculating circle through each triple of consecutive points.

    Uses the three-point circumradius formula R = abc / (4 * area),
    with Heron's formula for area. Works in any dimension because
    three points always span a plane.

    Large values indicate a nearly straight local path; small values
    indicate a sharp local bend. Collinear triples produce +inf.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)
    eps : float
        Area threshold below which the triple is treated as collinear.

    Returns
    -------
    ndarray of shape (n-2,)
    """
    n = len(trajectory)
    if n < 3:
        return np.zeros(0)
    a = np.linalg.norm(trajectory[1:n - 1] - trajectory[2:n], axis=1)
    b = np.linalg.norm(trajectory[0:n - 2] - trajectory[2:n], axis=1)
    c = np.linalg.norm(trajectory[0:n - 2] - trajectory[1:n - 1], axis=1)
    s = (a + b + c) / 2
    # Clamp Heron under the sqrt to guard against floating-point negatives.
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    radius = np.where(area > eps, (a * b * c) / (4 * np.maximum(area, eps)), np.inf)
    return radius


def velocity_covariance(trajectory, window=5):
    """Local outer-product structure tensor of velocity vectors.

    For each velocity step k, computes ``sum_i v_i v_i^T / count`` over
    the window of velocity vectors centered at k. The top eigenvector
    of the resulting (d, d) matrix is the dominant direction of local
    motion; the eigenvalue spectrum indicates how concentrated motion
    is along that direction (rank-1 means unidirectional flow; flat
    spectrum means diffuse motion).

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)
    window : int
        Half-width of the window. Each result[k] aggregates velocity
        vectors in indices [max(0, k-window), min(n_vel, k+window+1)).

    Returns
    -------
    ndarray of shape (n-1, d, d)
    """
    vel = np.diff(trajectory, axis=0)
    n_vel, d = vel.shape
    result = np.empty((n_vel, d, d))
    for k in range(n_vel):
        lo = max(0, k - window)
        hi = min(n_vel, k + window + 1)
        local = vel[lo:hi]
        result[k] = local.T @ local / max(len(local), 1)
    return result


# === Segmentation ===

def peaks(signal, threshold="auto", min_distance=1):
    """Find peak indices in a 1D signal.

    Used to find changepoints from angular_velocity or speed signals.

    When ``min_distance`` would exclude some candidates, the STRONGER
    peak wins: candidates are resolved greedily in descending magnitude,
    matching scipy/matlab conventions.

    Parameters
    ----------
    signal : ndarray of shape (n,)
        1D signal (e.g., angular velocity, speed, curvature magnitude).
    threshold : float or "auto"
        Minimum peak height. "auto" = mean + 1.5 * std (floored at 0.05).
    min_distance : int
        Minimum distance between returned peak indices.

    Returns
    -------
    list of int
        Indices of selected peaks, sorted ascending.
    """
    if len(signal) == 0:
        return []
    if threshold == "auto":
        threshold = _auto_threshold(signal, k=1.5)
    # Minimum absolute threshold floor.
    effective = max(threshold, 0.05)

    above = [i for i, v in enumerate(signal) if v > effective]
    if not above or min_distance <= 1:
        return above
    # Greedy-select by descending magnitude with min_distance constraint.
    above.sort(key=lambda i: signal[i], reverse=True)
    selected = []
    for i in above:
        if all(abs(i - j) >= min_distance for j in selected):
            selected.append(i)
    selected.sort()
    return selected


def segment(vectors, boundaries, meta=None):
    """Split a sequence at boundary indices.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    boundaries : list of int
        Split points (exclusive). E.g., [10, 25] splits into [0:10], [10:25], [25:n].
    meta : list of dicts, optional

    Returns
    -------
    list of dicts, each with:
        'vectors': ndarray, 'meta': list or None,
        'start': int, 'end': int
    """
    bounds = [0] + sorted(boundaries) + [len(vectors)]
    segments = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end <= start:
            continue
        segments.append({
            "vectors": vectors[start:end],
            "meta": meta[start:end] if meta else None,
            "start": start,
            "end": end,
        })
    return segments


def auto_segment(vectors, alpha=0.85, threshold="auto", meta=None,
                 recursive=True, min_segment_size=5,
                 window_size=None, sensitivity=1.0):
    """Detect changepoints and segment automatically.

    Combines smooth_exponential -> angular_velocity -> peaks -> segment.

    Two strategies are available, selected by ``threshold``:

    - Trajectory-based (default): uses an exponentially-smoothed
      trajectory and picks peaks in its angular velocity. If
      ``recursive=True`` (default), splits at the strongest peak and
      recurses into each half, keeping the detector sensitive even
      on long sequences.
    - Sliding window (``threshold="window"``): compares local means
      on each side of every candidate position. This path OVERRIDES
      ``recursive`` and does not use ``alpha``.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha : float
        Exponential smoothing factor. Ignored when threshold == "window".
    threshold : float, "auto", or "window"
        Peak detection threshold, or the literal "window" to select the
        sliding-window strategy.
    meta : list of dicts, optional
    recursive : bool
        If True, reset trajectory at changepoints and recurse. Ignored
        when threshold == "window".
    min_segment_size : int
        Minimum segment size (smaller segments are not further split).
    window_size : int or None
        Window size for the "window" method. None = auto (~10 items).
    sensitivity : float
        Threshold multiplier for the "window" method. Lower = more
        changepoints. Ignored unless threshold == "window".

    Returns
    -------
    list of segment dicts
    """
    if threshold == "window":
        boundaries = _window_changepoints(vectors, min_segment_size,
                                          window_size=window_size,
                                          sensitivity=sensitivity)
        return segment(vectors, boundaries, meta)
    elif recursive:
        boundaries = _recursive_changepoints(
            vectors, alpha, threshold, min_segment_size, offset=0
        )
        return segment(vectors, boundaries, meta)
    else:
        traj = smooth_exponential(vectors, alpha)
        angles = angular_velocity(traj)
        boundaries = [b + 1 for b in peaks(angles, threshold=threshold)]
        return segment(vectors, boundaries, meta)


def _window_changepoints(vectors, min_size=5, window_size=None, sensitivity=1.0):
    """Sliding-window changepoint detection.

    At each position j, compares the mean of a window before j to the
    mean of a window after j. High divergence = changepoint.

    This stays sensitive regardless of sequence length because it only
    looks at local context, not cumulative history.

    Parameters
    ----------
    window_size : int or None
        Number of items in each window. None = auto (max(10, min_size)).
    sensitivity : float
        Threshold multiplier on std. Lower = more changepoints.
    """
    n = len(vectors)
    if window_size is not None:
        window = window_size
    else:
        # Empirically: local mean stabilizes at ~10 items in 256-dim
        # embedding space. Fixed default, clamped to min_size.
        window = max(10, min_size)

    if n < window * 2:
        return []

    divergences = np.zeros(n)
    for j in range(window, n - window):
        mean_before = np.mean(vectors[j - window:j], axis=0)
        mean_after = np.mean(vectors[j:j + window], axis=0)
        if np.linalg.norm(mean_before) > 0 and np.linalg.norm(mean_after) > 0:
            divergences[j] = 1 - _cos_sim_scalar(
                _normalize(mean_before), _normalize(mean_after)
            )

    active = divergences[window:n - window]
    if len(active) == 0:
        return []
    thresh = max(_auto_threshold(active, k=sensitivity), 0.02)

    peak_indices = []
    for j in range(window, n - window):
        if divergences[j] >= thresh:
            # Local-maximum check.
            local = divergences[max(0, j - min_size):min(n, j + min_size + 1)]
            if divergences[j] >= np.max(local) - 1e-10:
                if not peak_indices or j - peak_indices[-1] >= min_size:
                    peak_indices.append(j)

    return peak_indices


def _recursive_changepoints(vectors, alpha, threshold, min_size, offset=0,
                            global_threshold=None):
    """Recursively detect changepoints with trajectory reset.

    Finds the strongest changepoint, splits, then recurses into each
    half. Returns absolute indices (adjusted by offset).

    Uses a global threshold computed from the full sequence on the first
    call, then applies it consistently to all sub-segments.
    """
    if len(vectors) < min_size * 2:
        return []

    traj = smooth_exponential(vectors, alpha)
    angles = angular_velocity(traj)

    if len(angles) == 0:
        return []

    # Compute global threshold on first call, reuse on recursion.
    # The k=1.0 multiplier is laxer than peaks()' k=1.5 because recursive
    # descent uses "at least one peak above threshold" as a split gate
    # and then picks the strongest, so a more permissive gate is right.
    if global_threshold is None:
        if threshold == "auto":
            global_threshold = _auto_threshold(angles, k=1.0)
        else:
            global_threshold = threshold

    peak_indices = [i for i, v in enumerate(angles)
                    if v > max(global_threshold, 0.02)]

    if not peak_indices:
        return []

    max_idx = max(peak_indices, key=lambda i: angles[i])
    cp = max_idx + 1

    if cp < min_size or len(vectors) - cp < min_size:
        return []

    left_cps = _recursive_changepoints(
        vectors[:cp], alpha, threshold, min_size,
        offset=offset, global_threshold=global_threshold
    )
    right_cps = _recursive_changepoints(
        vectors[cp:], alpha, threshold, min_size,
        offset=offset + cp, global_threshold=global_threshold
    )

    return left_cps + [offset + cp] + right_cps


# === Adaptive analysis ===

def adaptive_alpha(vectors, alpha_range=(0.3, 0.99), step=0.05):
    """Estimate the optimal alpha for a sequence.

    Finds the alpha that best predicts the next vector from the
    exponentially-smoothed trajectory. Measures the natural
    "memory length" of the sequence.

    On ties (e.g., a near-constant sequence where every alpha predicts
    perfectly), returns the grid value closest to the default 0.85.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha_range : tuple (low, high)
    step : float

    Returns
    -------
    float
        Optimal alpha, guaranteed within [alpha_range[0], alpha_range[1]].
    """
    if len(vectors) < 5:
        return 0.85

    # np.arange can overshoot the upper bound by a float epsilon; clamp.
    grid = np.arange(alpha_range[0], alpha_range[1] + step / 2, step)
    grid = np.minimum(grid, alpha_range[1])

    scores = np.empty(len(grid))
    for idx, alpha in enumerate(grid):
        traj = smooth_exponential(vectors, alpha)
        scores[idx] = np.mean([
            _cos_sim_scalar(traj[j], vectors[j + 1])
            for j in range(len(vectors) - 1)
        ])

    best = scores.max()
    tied = np.flatnonzero(scores >= best - 1e-9)
    # Tiebreak toward the default 0.85.
    idx = tied[np.argmin(np.abs(grid[tied] - 0.85))]
    return float(grid[idx])


def structural_richness(vectors, weight_fns=None, meta=None):
    """Inter-projection divergence: how much do different weightings disagree?

    Computes the mean pairwise cosine distance over a set of weighted
    projections of the sequence. High richness = position within the
    sequence matters (different weightings point in different
    directions). Low richness = semantically uniform.

    Cost note: each projection is a full fold over ``vectors``; the
    default set includes ``novelty_weights`` which runs an O(n*d) loop.
    On long sequences this compounds quickly.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    weight_fns : list of callables, optional
        Each callable takes ``(vectors, meta)`` and returns an (n,)
        weight array. Default:
            uniform, exponential(0.85), reverse_exponential(0.85), novelty.
    meta : list of dicts, optional

    Returns
    -------
    float
        Mean pairwise cosine distance between the weighted projections.
    """
    if len(vectors) == 0:
        raise ValueError("structural_richness requires at least one vector")
    if weight_fns is None:
        weight_fns = [
            lambda v, m: uniform_weights(len(v)),
            lambda v, m: exponential_weights(len(v), 0.85),
            lambda v, m: reverse_exponential_weights(len(v), 0.85),
            lambda v, m: novelty_weights(v),
        ]
    projections = np.array([
        weighted_mean(vectors, wf(vectors, meta)) for wf in weight_fns
    ])
    n_projs = len(projections)
    if n_projs < 2:
        return 0.0
    sim = _cosine_sim(projections)
    i, j = np.triu_indices(n_projs, k=1)
    return float(np.mean(1 - sim[i, j]))
