"""Weight generators and fold-style aggregation.

Each weight function returns an (n,) ndarray that can be passed to
``weighted_mean`` to fold a sequence into a single vector. Weights
compose under pointwise multiplication (numpy ``*``):

    w = exponential_weights(n, 0.85) * field_weights(meta, "role", {"user": 3.0})

This is a commutative, associative monoid with ``uniform_weights(n)``
(i.e., ``np.ones(n)``) as the identity element.
"""
import numpy as np


def uniform_weights(n):
    """Constant weights of 1. Identity for pointwise composition."""
    return np.ones(n)


def exponential_weights(n, alpha=0.85):
    """Recency weights: w_k = alpha^(n-1-k) for k=0..n-1.

    The lens convention w(j,k) = alpha^(k-j) evaluated at the final
    position: the last item has weight 1; older items decay
    geometrically. HIGHER alpha = LONGER memory; ``alpha=1`` degenerates
    to uniform weights (the running mean). Half-life: a weight halves
    every log(0.5)/log(alpha) steps (see ``alpha_to_half_life``).
    """
    return alpha ** np.arange(n - 1, -1, -1)


def reverse_exponential_weights(n, alpha=0.85):
    """Primacy weights: w_k = alpha^k for k=0..n-1.

    The first item has weight 1; newer items decay geometrically.
    Primacy mirror of the lens convention: HIGHER alpha = LONGER memory
    (more late items retained). Half-life log(0.5)/log(alpha) steps.
    """
    return alpha ** np.arange(n)


def gaussian_weights(n, focus=0.5, sigma=None):
    """Gaussian-focal weights peaked at ``focus``.

    Parameters
    ----------
    n : int
        Sequence length.
    focus : float or int
        Float in [0, 1] selects a fractional position (0=start, 1=end).
        Any other value is treated as an absolute index. To force
        absolute-index semantics, pass an int or a float outside [0, 1].
    sigma : float or None
        Standard deviation. ``None`` auto-selects ``max(n/6, 1)``.
    """
    if isinstance(focus, float) and 0.0 <= focus <= 1.0:
        k0 = focus * (n - 1)
    else:
        k0 = focus
    if sigma is None:
        sigma = max(n / 6, 1)
    k = np.arange(n)
    return np.exp(-((k - k0) ** 2) / (2 * sigma ** 2))


def time_decay_weights(times, half_life_seconds):
    """Exponential decay by real timestamp, anchored to ``max(times)``.

    Unlike ``exponential_weights`` (which decays by index), this decays
    by actual elapsed time. All weights lie in (0, 1]; the latest-
    timestamped item has weight 1.0. Non-monotone timestamps are
    handled by anchoring to ``max(times)``, not the final entry.

    Parameters
    ----------
    times : array-like of shape (n,)
        Timestamps as floats (epoch seconds) or anything castable.
    half_life_seconds : float
        Time for weight to halve.
    """
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return np.zeros(0)
    tau = half_life_seconds / np.log(2)
    return np.exp(-(times.max() - times) / tau)


def field_weights(meta, field, mapping=None, default=1.0):
    """Weight by a categorical or numeric metadata field.

    If ``mapping`` is provided, values are looked up and unmapped
    entries receive ``default``. Otherwise the field value is cast
    to float and used directly. Entries missing the field default to 1.0.

    Parameters
    ----------
    meta : list of dict
    field : str
    mapping : dict or None
    default : float
    """
    n = len(meta)
    w = np.ones(n)
    for k in range(n):
        val = meta[k].get(field)
        if mapping is not None:
            w[k] = mapping.get(val, default)
        elif val is not None:
            w[k] = float(val)
    return w


def novelty_weights(vectors):
    """Weight by deviation from the running mean.

    At each position, the weight is ``1 - cos(vectors[k], running_mean[0:k])``,
    clamped to a minimum of 0.01 so fully-aligned items never fall out
    of a composed projection entirely. ``w[0] = 1.0`` (no prior context).

    Because cosine similarity lies in [-1, 1], weights lie in [0.01, 2];
    anti-topical items receive values near 2.
    """
    n = len(vectors)
    w = np.ones(n)
    running_sum = np.zeros(vectors.shape[1])
    for k in range(n):
        if k > 0:
            mean = running_sum / k
            nm = np.linalg.norm(mean)
            nv = np.linalg.norm(vectors[k])
            if nm > 0 and nv > 0:
                sim = float((vectors[k] @ mean) / (nm * nv))
                w[k] = max(1 - sim, 0.01)
        running_sum += vectors[k]
    return w


def weighted_mean(vectors, weights=None):
    """Normalized weighted average of a sequence of vectors.

    Returns a unit-length vector of shape (d,). If ``weights`` is None,
    uses uniform weights.
    """
    if len(vectors) == 0:
        raise ValueError("weighted_mean requires at least one vector")
    if weights is None:
        weights = np.ones(len(vectors))
    m = np.average(vectors, axis=0, weights=weights)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m
