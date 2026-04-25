"""Path transforms: scan-style smoothing for trajectories in embedding space.

Each smoother takes an (n, d) sequence of vectors and returns an (n, d)
trajectory where each row is the unit-normalized weighted mean of the
prefix up to that position. These are the scan (running-projection)
counterparts of the fold-style reductions in ``weights``.

Invariant across all smoothers: ``smooth_x(vectors)[-1]`` equals
``weighted_mean(vectors, x_weights(len(vectors), ...))``.
"""
import numpy as np

from embflow.weights import (
    gaussian_weights,
    time_decay_weights,
    weighted_mean,
)


def _unit_rows(matrix):
    """Unit-normalize each row; zero-norm rows pass through unchanged."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=matrix.copy(), where=norms > 0)


def smooth_uniform(vectors):
    """Running mean: result[k] = unit(mean(vectors[0:k+1])).

    O(n) via cumulative sum.
    """
    if len(vectors) == 0:
        raise ValueError("smooth_uniform requires at least one vector")
    cum = np.cumsum(vectors, axis=0)
    means = cum / np.arange(1, len(vectors) + 1).reshape(-1, 1)
    return _unit_rows(means)


def smooth_exponential(vectors, alpha=0.85):
    """Exponential running projection: recency-weighted prefix mean.

    result[k] is the unit-normalized weighted mean of ``vectors[0:k+1]``
    with weights ``alpha^(k-j)`` for j=0..k.

    O(n) via the online recurrence
        num_k   = alpha * num_{k-1}   + vectors[k]
        denom_k = alpha * denom_{k-1} + 1
    so result[k] = unit(num_k / denom_k). Matches the old
    ``Exponential(alpha).trajectory(vectors)`` bit-for-bit.
    """
    if len(vectors) == 0:
        raise ValueError("smooth_exponential requires at least one vector")
    result = np.empty_like(vectors, dtype=float)
    num = np.zeros(vectors.shape[1])
    denom = 0.0
    for k in range(len(vectors)):
        num = alpha * num + vectors[k]
        denom = alpha * denom + 1.0
        m = num / denom
        nr = np.linalg.norm(m)
        result[k] = m / nr if nr > 0 else m
    return result


def smooth_reverse_exponential(vectors, alpha=0.85):
    """Primacy-weighted running projection: w_j = alpha^j.

    result[k] is the unit-normalized weighted mean of ``vectors[0:k+1]``
    with the older items receiving larger weights. O(n).
    """
    if len(vectors) == 0:
        raise ValueError("smooth_reverse_exponential requires at least one vector")
    result = np.empty_like(vectors, dtype=float)
    num = np.zeros(vectors.shape[1])
    denom = 0.0
    power = 1.0  # alpha^k, starts at 1
    for k in range(len(vectors)):
        num = num + power * vectors[k]
        denom = denom + power
        m = num / denom
        nr = np.linalg.norm(m)
        result[k] = m / nr if nr > 0 else m
        power *= alpha
    return result


def smooth_gaussian(vectors, focus=0.5, sigma=None):
    """Gaussian-focal running projection.

    O(n^2) by design: ``sigma`` defaults to ``max(n/6, 1)``, so weights
    depend on prefix length and cannot be precomputed once. Pass an
    explicit ``sigma`` for length-invariant focusing (still O(n^2) here
    since weights vary per prefix).
    """
    if len(vectors) == 0:
        raise ValueError("smooth_gaussian requires at least one vector")
    n = len(vectors)
    result = np.empty_like(vectors, dtype=float)
    for k in range(n):
        w = gaussian_weights(k + 1, focus=focus, sigma=sigma)
        result[k] = weighted_mean(vectors[:k + 1], w)
    return result


def smooth_time_decay(vectors, times, half_life_seconds):
    """Time-decay running projection anchored to max(times[:k+1]) at each step.

    O(n^2): the reference time changes with the prefix, so weights are
    recomputed per step.
    """
    if len(vectors) == 0:
        raise ValueError("smooth_time_decay requires at least one vector")
    n = len(vectors)
    times = np.asarray(times, dtype=float)
    result = np.empty_like(vectors, dtype=float)
    for k in range(n):
        w = time_decay_weights(times[:k + 1], half_life_seconds)
        result[k] = weighted_mean(vectors[:k + 1], w)
    return result
