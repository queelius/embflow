"""Null models: separate ORDER effects from COMPOSITION effects.

The methodological lesson of the embedding-dynamics experiments
(2026-06-10): composition alone induces structure in motion statistics.
Message eccentricity (distance from the conversation centroid) makes
both adjacent steps large regardless of order, so even the exchangeable
null has positive speed autocorrelation. Raw motion statistics conflate
composition and order; null-corrected statistics (real minus own-shuffle
null) separate them. Validated order effect sizes, real vs shuffled
(1,768 conversations, paired Cohen's d): speed -2.03, tortuosity +1.39,
adaptive alpha -1.83.

Turning-cosine lemma (package invariant, pinned by the test suite):
for exchangeable unit vectors with mean pairwise similarity mu,
E<v_k, v_{k+1}> = mu - 1 and E||v_k||^2 = 2 - 2*mu, so the expected
turning cosine is EXACTLY -1/2, independent of mu — i.e. independent of
anisotropy. Observed: -0.493 on 1,269 real conversations' shuffled user
sequences. Any motion statistic that survives shuffling is secretly
measuring content, not motion.
"""
import numpy as np

EPS = 1e-8


def shuffle(vectors, rng):
    """Full random permutation of the sequence. Returns a copy.

    The exchangeable null: destroys all order structure, preserves the
    multiset of vectors (and hence the mean embedding).

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    rng : numpy.random.Generator
    """
    return vectors[rng.permutation(len(vectors))]


def role_slot_shuffle(vectors, labels, rng):
    """Permute embeddings among same-label positions. Returns a copy.

    Preserves the label pattern, the per-label multiset, the mean
    embedding, and the per-label length profile; destroys only content
    ORDER. The primary null for role-alternating sequences (e.g.
    user/assistant conversations), where a full shuffle would also
    destroy the role-alternation zigzag.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    labels : array-like of shape (n,)
        Group key per position (e.g. "user"/"assistant").
    rng : numpy.random.Generator
    """
    labels = np.asarray(labels)
    out = vectors.copy()
    for r in np.unique(labels):
        idx = np.where(labels == r)[0]
        out[idx] = vectors[idx[rng.permutation(len(idx))]]
    return out


def null_corrected(stat_fn, vectors, labels=None, K=5, seed=42):
    """Evaluate a statistic against its own shuffle null.

    Computes ``real = stat_fn(vectors)`` and the mean of ``stat_fn`` over
    ``K`` shuffles (role-slot shuffle when ``labels`` is given, full
    shuffle otherwise), and returns ``(real, null_mean, real - null_mean)``.
    The difference is the ORDER effect; the null mean is the composition
    floor. ``stat_fn`` may return a float or a dict of floats (e.g.
    ``motion_signature``); dicts are corrected per key. NaN null draws
    are dropped per key via nanmean (all-NaN propagates NaN).

    Parameters
    ----------
    stat_fn : callable (ndarray (n, d)) -> float or dict[str, float]
    vectors : ndarray of shape (n, d)
    labels : array-like of shape (n,), optional
        Switches the null from full shuffle to role-slot shuffle.
    K : int
        Number of shuffles (default 5, the validated setting).
    seed : int
        Seed for the shuffle stream (default 42, the experiments' seed).

    Returns
    -------
    (real, null_mean, diff) — floats, or dicts keyed like ``stat_fn``'s.
    """
    rng = np.random.default_rng(seed)
    real = stat_fn(vectors)
    nulls = []
    for _ in range(K):
        if labels is not None:
            shuffled = role_slot_shuffle(vectors, labels, rng)
        else:
            shuffled = shuffle(vectors, rng)
        nulls.append(stat_fn(shuffled))
    if isinstance(real, dict):
        null_mean = {k: float(np.nanmean([n[k] for n in nulls])) for k in real}
        diff = {k: real[k] - null_mean[k] for k in real}
        return real, null_mean, diff
    real = float(real)
    null_mean = float(np.nanmean(nulls))
    return real, null_mean, real - null_mean


def paired_stats(real, shuffled):
    """Cohort-level paired comparison real vs shuffled (rows aligned).

    NaN pairs are dropped. ``cohens_d_paired`` is mean(diff)/std(diff):
    the effect-size convention behind the validated numbers quoted in
    this module's docstring.

    Parameters
    ----------
    real, shuffled : ndarray of shape (m,)
        One statistic per sequence, e.g. ``tortuosity`` per conversation
        and its own-shuffle null mean.

    Returns
    -------
    dict with n, mean_real, mean_shuffled, mean_diff, cohens_d_paired,
    frac_positive_diff.
    """
    real = np.asarray(real, dtype=float)
    shuffled = np.asarray(shuffled, dtype=float)
    ok = ~(np.isnan(real) | np.isnan(shuffled))
    r, s = real[ok], shuffled[ok]
    diff = r - s
    return {
        "n": int(ok.sum()),
        "mean_real": float(r.mean()),
        "mean_shuffled": float(s.mean()),
        "mean_diff": float(diff.mean()),
        "cohens_d_paired": float(diff.mean() / (diff.std() + EPS)),
        "frac_positive_diff": float((diff > 0).mean()),
    }
