"""Trajectory comparison: distance and similarity between embedding sequences.

Multiple comparison methods for sequences of different lengths,
operating in semantic space, transition space, or shape space.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine_dist(a, b):
    """Cosine distance between two vectors."""
    return 1 - float(_cosine_sim(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def _resample(traj, k):
    """Resample trajectory to k evenly spaced points."""
    idx = np.linspace(0, len(traj) - 1, k).astype(int)
    return traj[idx]


def trajectory_distance(traj_a, traj_b, method="dtw", **kwargs):
    """Compute distance between two trajectories.

    Parameters
    ----------
    traj_a, traj_b : ndarray of shape (n, d) and (m, d)
        Two trajectories (can be different lengths).
    method : str
        'dtw': Dynamic Time Warping (optimal alignment)
        'resample': Resample to same length, mean pointwise distance
        'frechet': Frechet distance (worst-case leash length)
        'shape': Compare curvature profiles via DTW (shape-only)
        'endpoint': Distance between start/end pairs

    Returns
    -------
    float
        Distance (0 = identical, higher = more different).
    """
    if len(traj_a) == 0 or len(traj_b) == 0:
        raise ValueError("trajectory_distance requires non-empty trajectories")
    if method == "dtw":
        return _dtw_distance(traj_a, traj_b)
    elif method == "resample":
        k = kwargs.get("k", 20)
        return _resample_distance(traj_a, traj_b, k)
    elif method == "frechet":
        return _frechet_distance(traj_a, traj_b)
    elif method == "shape":
        return _shape_distance(traj_a, traj_b)
    elif method == "endpoint":
        return _endpoint_distance(traj_a, traj_b)
    else:
        raise ValueError(f"Unknown method: {method}")


def _dtw_distance(a, b):
    """Dynamic Time Warping distance using cosine distance.

    Normalized by path length (n + m). When one input has a single
    point, the only valid warping path has max(n, m) steps and the
    result is effectively the mean cosine distance from that point
    to every point in the longer sequence.
    """
    n, m = len(a), len(b)
    # Cost matrix
    cost = np.full((n, m), np.inf)
    cost[0, 0] = _cosine_dist(a[0], b[0])

    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + _cosine_dist(a[i], b[0])
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + _cosine_dist(a[0], b[j])
    for i in range(1, n):
        for j in range(1, m):
            d = _cosine_dist(a[i], b[j])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[n - 1, m - 1] / (n + m)  # normalize by path length


def _resample_distance(a, b, k=20):
    """Resample both to k points, mean cosine distance."""
    ra = _resample(a, k)
    rb = _resample(b, k)
    dists = [_cosine_dist(ra[i], rb[i]) for i in range(k)]
    return float(np.mean(dists))


def _frechet_distance(a, b):
    """Discrete Frechet distance (minimax leash length).

    Iterative bottom-up DP; dp[i, j] is the minimum over all monotone
    coupling paths ending at (i, j) of the max cosine distance on the path.
    """
    n, m = len(a), len(b)
    dp = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            d = _cosine_dist(a[i], b[j])
            if i == 0 and j == 0:
                dp[i, j] = d
            elif i == 0:
                dp[i, j] = max(dp[0, j - 1], d)
            elif j == 0:
                dp[i, j] = max(dp[i - 1, 0], d)
            else:
                dp[i, j] = max(min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]), d)

    return dp[n - 1, m - 1]


def _shape_distance(a, b):
    """Shape distance: compare curvature magnitude profiles via DTW.

    Ignores where in embedding space the trajectories live.
    Only compares the shape of the path (where it turns, how sharply).
    """
    from embflow.ops import angular_velocity

    if len(a) < 3 or len(b) < 3:
        return 1.0

    # Angular velocity as 1D signal
    av_a = angular_velocity(a).reshape(-1, 1)
    av_b = angular_velocity(b).reshape(-1, 1)

    return _dtw_distance(av_a, av_b)


def _endpoint_distance(a, b):
    """Mean of start-start and end-end distances."""
    d_start = _cosine_dist(a[0], b[0])
    d_end = _cosine_dist(a[-1], b[-1])
    return (d_start + d_end) / 2


def continuation_score(vectors_a, vectors_b, alpha=0.85):
    """Score how well sequence B continues from sequence A.

    Compares the recency-weighted end of A with the primacy-weighted
    start of B. High score = B picks up where A left off.

    Parameters
    ----------
    vectors_a, vectors_b : ndarray of shape (n, d) and (m, d)
    alpha : float

    Returns
    -------
    float
        Cosine similarity (-1 to 1, higher = stronger continuation).
    """
    from embflow.weights import (
        exponential_weights,
        reverse_exponential_weights,
        weighted_mean,
    )

    if len(vectors_a) == 0 or len(vectors_b) == 0:
        raise ValueError("continuation_score requires non-empty sequences")
    end_a = weighted_mean(vectors_a, exponential_weights(len(vectors_a), alpha))
    start_b = weighted_mean(
        vectors_b, reverse_exponential_weights(len(vectors_b), alpha)
    )
    return 1 - _cosine_dist(end_a, start_b)
