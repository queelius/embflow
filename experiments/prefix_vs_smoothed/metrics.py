"""Metrics comparing a candidate trajectory to the prefix trajectory.

Three families of metrics:

1. Pointwise: cosine similarity at each k.
2. Trajectory: DTW and shape distance using embflow's own machinery.
3. Downstream: do scalar operators agree? does auto_segment agree?
   does either path recover known ground-truth changepoints (synthetic)?

The downstream metrics are arguably the most important: even if pointwise
divergence is large, the candidate is "good enough" if the operators and
segmentation behave the same way the prefix path's would.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import embflow as ef


def pointwise_cosine(prefix, candidate):
    """Cosine similarity at each k. Inputs must be (n, d) and unit-normalized."""
    n = len(prefix)
    sims = np.empty(n)
    for k in range(n):
        sims[k] = float(cosine_similarity(prefix[k:k + 1], candidate[k:k + 1])[0, 0])
    return sims


def trajectory_dtw(prefix, candidate):
    # Clamp tiny negative values from floating-point noise to 0.
    return max(float(ef.trajectory_distance(prefix, candidate, method="dtw")), 0.0)


def trajectory_shape(prefix, candidate):
    if len(prefix) < 3 or len(candidate) < 3:
        return None
    return float(ef.trajectory_distance(prefix, candidate, method="shape"))


def operator_agreement(prefix, candidate):
    """Pearson correlation of scalar operators along the trajectory."""
    out = {
        "angular_velocity": _safe_corr(
            ef.angular_velocity(prefix), ef.angular_velocity(candidate)
        ),
        "arc_length": _safe_corr(
            ef.arc_length(prefix), ef.arc_length(candidate)
        ),
        "local_curvature_radius": _curvature_radius_corr(prefix, candidate),
    }
    return out


def _curvature_radius_corr(prefix, candidate):
    if len(prefix) < 3 or len(candidate) < 3:
        return None
    rp = ef.local_curvature_radius(prefix)
    rc = ef.local_curvature_radius(candidate)
    finite = np.isfinite(rp) & np.isfinite(rc)
    if finite.sum() < 2:
        return None
    return _safe_corr(rp[finite], rc[finite])


def _safe_corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def segmentation_agreement(prefix, candidate, alpha=0.85, tolerance=1,
                           min_segment_size=5):
    """Jaccard agreement of changepoints between prefix and candidate paths.

    A changepoint from one path is considered matched if it lies within
    ``tolerance`` of a changepoint from the other.

    Returns:
        jaccard       : float in [0, 1] (1.0 if both paths agree on empty)
        n_prefix      : int, number of changepoints in the prefix path
        n_candidate   : int, number of changepoints in the candidate path

    The counts let the caller distinguish "vacuous agreement on empty"
    from "real agreement on detected boundaries."
    """
    bnd_prefix = _boundaries(
        ef.auto_segment(prefix, alpha=alpha, min_segment_size=min_segment_size)
    )
    bnd_cand = _boundaries(
        ef.auto_segment(candidate, alpha=alpha, min_segment_size=min_segment_size)
    )
    return (
        _jaccard_within_tolerance(bnd_prefix, bnd_cand, tolerance),
        len(bnd_prefix),
        len(bnd_cand),
    )


def changepoint_recall(found, ground_truth, tolerance=1):
    """Fraction of ground-truth changepoints recovered by the segmenter.

    Each GT changepoint is matched if some found changepoint is within
    ``tolerance``. ``found`` is a list of segments (from auto_segment);
    ``ground_truth`` is a list of int changepoint indices.
    """
    found_idx = _boundaries(found)
    if not ground_truth:
        return 1.0 if not found_idx else 0.0
    matched = sum(
        any(abs(g - f) <= tolerance for f in found_idx)
        for g in ground_truth
    )
    return matched / len(ground_truth)


def _boundaries(segments):
    return [s["start"] for s in segments if s["start"] > 0]


def _jaccard_within_tolerance(a, b, tolerance):
    a_set, b_set = set(a), set(b)
    if not a_set and not b_set:
        return 1.0
    matched_a = sum(any(abs(x - y) <= tolerance for y in b_set) for x in a_set)
    union = len(a_set) + len(b_set) - matched_a
    return matched_a / union if union else 1.0
