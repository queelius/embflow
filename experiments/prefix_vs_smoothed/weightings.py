"""Candidate weighting schemes for combining per-item embeddings.

Each function returns a candidate trajectory of shape (n, d), with rows
unit-normalized. The candidates are compared against the prefix path
to measure how well a linear combination of per-item embeddings can
reproduce the genuinely-prefix-embedded trajectory.
"""
import numpy as np

import embflow as ef

from experiments.prefix_vs_smoothed.paths import normalize_rows


def candidate_uniform(per_item):
    """Running mean: w_kj = 1 for all j <= k."""
    return ef.smooth_uniform(per_item)


def candidate_length_weighted(per_item, lengths):
    """Length-weighted running mean.

    For TF-IDF embedders this is the analytically correct weighting,
    so the resulting trajectory should match the prefix path almost
    exactly. Use it as a positive control.
    """
    n = per_item.shape[0]
    result = np.empty_like(per_item, dtype=float)
    weights = np.asarray(lengths, dtype=float)
    for k in range(n):
        w = weights[:k + 1]
        if w.sum() == 0:
            w = np.ones(k + 1)
        result[k] = ef.weighted_mean(per_item[:k + 1], w)
    return result


def candidate_exponential(per_item, alpha):
    """Exponential running projection from embflow.smooth (O(n))."""
    return ef.smooth_exponential(per_item, alpha)


def candidate_per_conversation_lstsq(per_item, prefix):
    """Best-possible per-conversation linear fit.

    For each k, solve ``min_w || A w - prefix[k] ||`` where the columns
    of A are the per-item embeddings up to position k. This is an
    upper bound on what any linear weighting can recover; if the
    parameterized candidates fall far below this ceiling, there's room
    to design a better fixed weighting.

    Per-conversation least squares is overfitting in the strict sense,
    but the goal here is "what is the best a linear combination *could*
    do," not "is this generalizable."
    """
    n = per_item.shape[0]
    result = np.empty_like(per_item, dtype=float)
    for k in range(n):
        A = per_item[:k + 1].T            # (d, k+1)
        b = prefix[k]                      # (d,)
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        v = A @ w
        nv = np.linalg.norm(v)
        result[k] = v / nv if nv > 0 else v
    return result


def all_candidates(per_item, lengths, prefix=None,
                   alpha_grid=(0.5, 0.7, 0.85, 0.95, 0.99),
                   include_lstsq=True):
    """Run every candidate scheme. Returns dict {name -> (n, d) trajectory}.

    ``prefix`` is needed only for the per-conversation least-squares
    ceiling; pass None to skip it.
    """
    candidates = {}
    candidates["uniform"] = candidate_uniform(per_item)
    candidates["length_weighted"] = candidate_length_weighted(per_item, lengths)
    for alpha in alpha_grid:
        key = f"exponential(alpha={alpha:.2f})"
        candidates[key] = candidate_exponential(per_item, alpha)
    if include_lstsq and prefix is not None:
        candidates["per_conv_lstsq_ceiling"] = candidate_per_conversation_lstsq(
            per_item, prefix
        )
    # All candidates are unit-row-normalized for cosine comparability.
    return {name: normalize_rows(traj) for name, traj in candidates.items()}
