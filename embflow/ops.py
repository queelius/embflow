"""Sequence operations: derivatives, segmentation, and analysis.

These operate on sequences of vectors (typically trajectories produced
by lens.trajectory(), but also raw embedding sequences).
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def velocity(trajectory):
    """First derivative: direction and magnitude of change at each step.

    For a trajectory t_0, t_1, ..., t_{n-1}, returns:
        v_k = t_{k+1} - t_k   for k = 0..n-2

    The velocity vectors live in "transition space": similarity between
    velocity vectors means "changing in the same way," independent of
    what the embeddings are about.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-1, d)
    """
    return np.diff(trajectory, axis=0)


def curvature(trajectory):
    """Second derivative: rate of change of direction.

    High curvature = the trajectory is turning sharply (changepoint).

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-2, d)
    """
    return np.diff(trajectory, n=2, axis=0)


def speed(trajectory):
    """Scalar speed at each step: magnitude of velocity.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n-1,)
    """
    vel = velocity(trajectory)
    return np.linalg.norm(vel, axis=1)


def angular_velocity(trajectory):
    """Angular change between consecutive steps.

    Measures the cosine distance between consecutive trajectory points.
    Equivalent to 1 - cos(t_k, t_{k+1}).

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
        sim = float(_cosine_sim(
            trajectory[k].reshape(1, -1),
            trajectory[k + 1].reshape(1, -1)
        )[0, 0])
        angles[k] = 1 - sim
    return angles


def drift(trajectory):
    """Total semantic drift: cosine distance from first to last point.

    Parameters
    ----------
    trajectory : ndarray of shape (n, d)

    Returns
    -------
    float
    """
    if len(trajectory) < 2:
        return 0.0
    return 1 - float(_cosine_sim(
        trajectory[0].reshape(1, -1),
        trajectory[-1].reshape(1, -1)
    )[0, 0])


def peaks(signal, threshold="auto", min_distance=1):
    """Find peak indices in a 1D signal.

    Used to find changepoints from angular_velocity or speed signals.

    Parameters
    ----------
    signal : ndarray of shape (n,)
        1D signal (e.g., angular velocity, speed, curvature magnitude).
    threshold : float or "auto"
        Minimum peak height. "auto" = mean + 1.5 * std.
    min_distance : int
        Minimum distance between peaks.

    Returns
    -------
    list of int
        Indices of peaks.
    """
    if len(signal) == 0:
        return []
    if threshold == "auto":
        threshold = np.mean(signal) + 1.5 * np.std(signal)
    min_abs = 0.05  # minimum absolute threshold

    candidates = [i for i, v in enumerate(signal) if v > max(threshold, min_abs)]

    # Enforce minimum distance
    if min_distance > 1 and candidates:
        filtered = [candidates[0]]
        for c in candidates[1:]:
            if c - filtered[-1] >= min_distance:
                filtered.append(c)
        candidates = filtered

    return candidates


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
        seg = {
            "vectors": vectors[start:end],
            "meta": meta[start:end] if meta else None,
            "start": start,
            "end": end,
        }
        segments.append(seg)
    return segments


def auto_segment(vectors, alpha=0.85, threshold="auto", meta=None):
    """Detect changepoints and segment automatically.

    Combines trajectory -> angular_velocity -> peaks -> segment.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha : float
        Exponential decay for trajectory computation.
    threshold : float or "auto"
        Peak detection threshold.
    meta : list of dicts, optional

    Returns
    -------
    list of segment dicts
    """
    from embflow.lens import Exponential
    traj = Exponential(alpha).trajectory(vectors)
    angles = angular_velocity(traj)
    boundaries = peaks(angles, threshold=threshold)
    # Shift by 1 because angular_velocity[k] corresponds to the gap
    # between trajectory[k] and trajectory[k+1], so the changepoint
    # is at position k+1 in the original sequence
    boundaries = [b + 1 for b in boundaries]
    return segment(vectors, boundaries, meta)


def adaptive_alpha(vectors, alpha_range=(0.3, 0.99), step=0.05):
    """Estimate the optimal alpha for a sequence.

    Finds the alpha that best predicts the next vector from the
    exponential trajectory. This measures the natural "memory length"
    of the sequence.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha_range : tuple (low, high)
    step : float

    Returns
    -------
    float
        Optimal alpha.
    """
    from embflow.lens import Exponential

    if len(vectors) < 5:
        return 0.85

    best_alpha = 0.85
    best_score = -1

    for alpha in np.arange(alpha_range[0], alpha_range[1] + step / 2, step):
        traj = Exponential(alpha).trajectory(vectors)
        # How well does traj[j] predict vectors[j+1]?
        pred_sims = []
        for j in range(len(vectors) - 1):
            s = float(_cosine_sim(
                traj[j].reshape(1, -1),
                vectors[j + 1].reshape(1, -1)
            )[0, 0])
            pred_sims.append(s)
        score = np.mean(pred_sims)
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return float(best_alpha)


def structural_richness(vectors, lenses=None, meta=None):
    """Inter-lens divergence: how much do different lenses disagree?

    High richness = the conversation's structure matters (start, end,
    surprises all point in different directions).
    Low richness = semantically uniform (all lenses agree).

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    lenses : list of Lens, optional
        Default: [Uniform, Exponential(0.85), ReverseExponential(0.85), Surprise]
    meta : list of dicts, optional

    Returns
    -------
    float
        Mean pairwise cosine distance between lens projections.
    """
    from embflow.lens import Uniform, Exponential, ReverseExponential, Surprise

    if lenses is None:
        lenses = [Uniform(), Exponential(0.85), ReverseExponential(0.85), Surprise()]

    projections = np.array([lens.project(vectors, meta) for lens in lenses])
    sim = _cosine_sim(projections)
    n = len(lenses)
    # Mean pairwise distance
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (1 - sim[i, j])
            count += 1
    return total / count if count > 0 else 0.0
