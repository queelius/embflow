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


def auto_segment(vectors, alpha=0.85, threshold="auto", meta=None,
                 recursive=True, min_segment_size=5,
                 window_size=None, sensitivity=1.0):
    """Detect changepoints and segment automatically.

    Combines trajectory -> angular_velocity -> peaks -> segment.

    When recursive=True (default), resets the trajectory at each
    changepoint and recursively detects sub-changepoints within
    each segment. This keeps the detector sensitive in long sequences
    instead of letting the exponential smoothing absorb all variation.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha : float
        Exponential decay for trajectory computation.
    threshold : float, "auto", or "window"
        Peak detection threshold. "window" uses sliding window method.
    meta : list of dicts, optional
    recursive : bool
        If True, reset trajectory at changepoints and recurse.
    min_segment_size : int
        Minimum segment size (won't split segments smaller than this).
    window_size : int or None
        Window size for "window" method. None = auto.
    sensitivity : float
        Threshold multiplier for "window" method. Lower = more changepoints.

    Returns
    -------
    list of segment dicts
    """
    if isinstance(threshold, str) and threshold == "window":
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
        from embflow.lens import Exponential
        traj = Exponential(alpha).trajectory(vectors)
        angles = angular_velocity(traj)
        boundaries = peaks(angles, threshold=threshold)
        boundaries = [b + 1 for b in boundaries]
        return segment(vectors, boundaries, meta)


def _window_changepoints(vectors, min_size=5, window_size=None, sensitivity=1.0):
    """Sliding window changepoint detection.

    At each position j, compares the mean of a window before j to the
    mean of a window after j. High divergence = changepoint.

    This stays sensitive regardless of conversation length because
    it only looks at local context, not cumulative history.

    Parameters
    ----------
    window_size : int or None
        Number of messages in each window. None = auto (min(n//8, 20)).
    sensitivity : float
        Threshold multiplier on std. Lower = more changepoints.
    """
    n = len(vectors)
    if window_size is not None:
        window = window_size
    else:
        # Empirically: local mean stabilizes at ~10 messages in 256-dim
        # embedding space. Use this as fixed default, clamped to min_size.
        window = max(10, min_size)

    if n < window * 2:
        return []

    divergences = np.zeros(n)
    for j in range(window, n - window):
        before = vectors[j - window:j]
        after = vectors[j:j + window]
        mean_before = np.mean(before, axis=0)
        mean_after = np.mean(after, axis=0)
        nb = np.linalg.norm(mean_before)
        na = np.linalg.norm(mean_after)
        if nb > 0 and na > 0:
            sim = float(_cosine_sim(
                (mean_before / nb).reshape(1, -1),
                (mean_after / na).reshape(1, -1)
            )[0, 0])
            divergences[j] = 1 - sim

    # Find peaks above threshold
    active = divergences[window:n - window]
    if len(active) == 0:
        return []
    thresh = np.mean(active) + sensitivity * np.std(active)
    thresh = max(thresh, 0.02)

    peak_indices = []
    for j in range(window, n - window):
        if divergences[j] >= thresh:
            # Local maximum check
            local = divergences[max(0, j - min_size):min(n, j + min_size + 1)]
            if divergences[j] >= np.max(local) - 1e-10:
                if not peak_indices or j - peak_indices[-1] >= min_size:
                    peak_indices.append(j)

    return peak_indices


def _recursive_changepoints(vectors, alpha, threshold, min_size, offset=0,
                            global_threshold=None):
    """Recursively detect changepoints with trajectory reset.

    Finds the strongest changepoint, splits, then recurses into
    each half. Returns absolute indices (adjusted by offset).

    Uses a global threshold computed from the full sequence on the
    first call, then applies it consistently to all sub-segments.
    """
    from embflow.lens import Exponential

    if len(vectors) < min_size * 2:
        return []

    traj = Exponential(alpha).trajectory(vectors)
    angles = angular_velocity(traj)

    if len(angles) == 0:
        return []

    # Compute global threshold on first call, reuse on recursion
    if global_threshold is None:
        if threshold == "auto":
            global_threshold = np.mean(angles) + 1.0 * np.std(angles)
        else:
            global_threshold = threshold

    # Find all peaks above the global threshold
    peak_indices = [i for i, v in enumerate(angles)
                    if v > max(global_threshold, 0.02)]

    if not peak_indices:
        return []

    # Take the strongest peak
    max_idx = max(peak_indices, key=lambda i: angles[i])
    cp = max_idx + 1

    if cp < min_size or len(vectors) - cp < min_size:
        return []

    # Recurse with the same global threshold
    left_cps = _recursive_changepoints(
        vectors[:cp], alpha, threshold, min_size,
        offset=offset, global_threshold=global_threshold
    )
    right_cps = _recursive_changepoints(
        vectors[cp:], alpha, threshold, min_size,
        offset=offset + cp, global_threshold=global_threshold
    )

    return left_cps + [offset + cp] + right_cps


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
