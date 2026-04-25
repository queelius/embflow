# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`embflow` is a small Python library (~1200 LOC, alpha) that treats an ordered
sequence of embedding vectors as a path in R^d and provides a calculus over
such paths: weighted folds, scan-style smoothing, differential operators,
second-order geometry, segmentation, and distance. It is embedding-source-
agnostic: any producer of vectors is upstream, embflow is downstream.
`numpy` and `scikit-learn` (only for `cosine_similarity`) are the sole
runtime deps.

The v0.2.0 rewrite removed the `Lens` class hierarchy and replaced it with
plain functions (weight generators + smoothers). The organizing concept is
now the trajectory itself, not the projection onto it.

## Commands

```bash
pip install -e ".[dev]"                    # editable install with pytest + pytest-cov
pytest                                     # full test suite (single file: tests/test_core.py)
pytest tests/test_core.py::TestWeightComposition::test_multiplication_is_commutative  # one test
pytest --cov=embflow --cov-report=term-missing                                         # coverage
python -m build                            # build sdist/wheel (version from embflow.__version__)
```

`pyproject.toml` pulls `version` dynamically from `embflow/__init__.py`, so
bump `__version__` there when cutting a release, not in `pyproject.toml`.

## Architecture

Four source modules organized around a clear axis. All public names are
re-exported from `embflow/__init__.py`; add new symbols there or external
callers will not see them.

### `weights.py`: fold-style primitives

Weight generators return an `(n,)` ndarray. `weighted_mean(vectors, weights)`
folds a sequence to a single unit-normalized vector.

Generators: `uniform_weights(n)`, `exponential_weights(n, alpha)` (recency),
`reverse_exponential_weights(n, alpha)` (primacy), `gaussian_weights(n, focus, sigma)`
(focal), `time_decay_weights(times, half_life_seconds)` (real-timestamp decay
anchored to `max(times)`), `field_weights(meta, field, mapping)` (metadata-
dispatched), `novelty_weights(vectors)` (deviation from running mean).

Weights form a **commutative, associative monoid under pointwise
multiplication** (`*`), with `uniform_weights(n)` as the identity. That
structure is inherited directly from numpy, so composition happens with
plain `*`:

```python
w = exponential_weights(n, 0.85) * field_weights(meta, "role", {"user": 3.0})
projection = weighted_mean(vectors, w)
```

The monoid properties (commutativity, associativity, identity) are pinned
by `TestWeightComposition`. Breaking any is an API break.

### `smooth.py`: scan-style primitives

Each smoother takes an `(n, d)` sequence and returns an `(n, d)` trajectory
where row `k` is the unit-normalized weighted mean of the prefix `[0..k]`.

Invariant across all smoothers: `smooth_x(vectors)[-1] == weighted_mean(vectors, x_weights(n, ...))`.

- `smooth_uniform(vectors)`: running mean via cumulative sum, O(n).
- `smooth_exponential(vectors, alpha)`: exponential running projection via
  the online recurrence `num_k = alpha*num_{k-1} + vectors[k]`,
  `denom_k = alpha*denom_{k-1} + 1`. O(n). Matches the formula
  `sum_j alpha^(k-j) vectors[j] / sum_j alpha^(k-j)` at every prefix, then
  unit-normalized.
- `smooth_reverse_exponential(vectors, alpha)`: primacy-weighted, O(n).
- `smooth_gaussian(vectors, focus, sigma)`: O(n^2) because auto-sigma
  depends on prefix length. Pass explicit `sigma` for length-invariant
  focusing (still O(n^2) here since weights vary per prefix).
- `smooth_time_decay(vectors, times, half_life_seconds)`: O(n^2); the
  reference time changes with each prefix.

### `ops.py`: differential and geometric operators

All operate on `(n, d)` arrays. These are the library's most distinctive
surface: they treat the sequence as a path and apply calculus-of-curves
operators to it.

**First-order**: `velocity` (first difference), `speed` (|velocity|),
`angular_velocity` (1 - cos between adjacent trajectory points).

**Second-order**: `curvature` (second difference), `local_curvature_radius`
(three-point circumradius via Heron, `abc / (4*area)`, works in any
dimension because three points span a plane; collinear triples give +inf),
`velocity_covariance` (local outer-product structure tensor over a window
of velocity vectors; top eigenvector = dominant flow direction, eigenvalue
spectrum = anisotropy).

**Third-order**: `jerk` (third difference).

**Globals**: `arc_length` (cumulative Euclidean segment lengths),
`drift` (cosine distance from first to last).

**Segmentation**: `peaks(signal, threshold="auto", min_distance=N)` returns
sorted indices. When `min_distance` excludes candidates, the STRONGER peak
wins (greedy descending-magnitude selection, then re-sort by index). This
is scipy-style semantics, not first-come.

`auto_segment(vectors, alpha, threshold=...)` has two strategies:
- Trajectory-based (default): `smooth_exponential -> angular_velocity ->
  peaks -> segment`. With `recursive=True` (default), splits at the
  strongest peak and recurses into each half under a single global
  threshold (`_recursive_changepoints`).
- Sliding-window (`threshold="window"`): compares mean-before to mean-after
  at each position (`_window_changepoints`). Stays sensitive on long
  sequences. OVERRIDES `recursive` and does not use `alpha`.

**Adaptive analysis**: `adaptive_alpha` grid-searches `alpha` by one-step
predictive cosine similarity from `smooth_exponential(vectors, alpha)[j]`
to `vectors[j+1]`. O(n * grid_size) now that `smooth_exponential` is O(n).
On ties (near-constant input), returns the alpha closest to 0.85. The grid
is clamped to `alpha_range[1]` to avoid `np.arange` float overshoot.

`structural_richness(vectors, weight_fns=None)` computes the mean pairwise
cosine distance over a set of weighted projections. The default set is
`[uniform, exponential(0.85), reverse_exponential(0.85), novelty]`. Users
can pass a list of `(vectors, meta) -> (n,)` callables for custom
viewpoints.

Shared `_auto_threshold(signal, k=1.5)` helper is used by both `peaks()`
and `_recursive_changepoints`. The latter uses `k=1.0` deliberately: it's
a split-gate (any peak above threshold allows a split, then the strongest
is chosen), so a laxer multiplier is correct there.

### `compare.py`: trajectory distances

`trajectory_distance(a, b, method=...)` operates in different geometric
spaces. Empty inputs raise `ValueError` at entry.

- `"dtw"` / `"resample"` / `"frechet"`: **semantic space**, compare where
  the trajectories are. `"dtw"` and `"frechet"` handle different lengths
  natively; `"resample"` forces `k` common points. For a 1-element
  trajectory vs. an m-element one, DTW degenerates to mean cosine distance
  (documented in `_dtw_distance`).
- `"shape"`: **shape space**, DTW on angular-velocity profiles. Matches
  trajectories with the same bend pattern even in unrelated regions of
  embedding space. Returns 1.0 for trajectories shorter than 3.
- `"endpoint"`: mean of start-start and end-end cosine distances.

`continuation_score(a, b, alpha)` uses the new weight primitives:
`weighted_mean(a, exponential_weights(len(a), alpha))` (recency end of A)
compared to `weighted_mean(b, reverse_exponential_weights(len(b), alpha))`
(primacy start of B). "Does B pick up where A left off."

## Design conventions

- **All projections normalize to unit length.** `_normalize` (on vectors)
  is duplicated per module; each copy short-circuits on zero-norm.
  Don't consolidate into a shared utility without preserving the
  zero-vector handling.
- **Weight composition is just numpy `*`.** If you add a new weight
  generator, return a plain `(n,)` ndarray. No wrapper classes.
- **Metadata is schema-less `list[dict]`.** `field_weights` uses `.get()`
  and treats missing entries as weight 1.0. There is no implicit
  "no-meta falls back to uniform" like the old `Lens` had; the caller
  must guard if they might have `meta=None`.
- **Scalar similarity goes through `sklearn.metrics.pairwise.cosine_similarity`.**
  Overkill for 1-to-1 cosines, but it's the only vectorization point in
  play. Consistency over micro-optimization.
- **Public API comes from `__init__.py` re-exports.** Add new symbols
  there or external callers will not see them.
- **O(n) when online, O(n^2) when weights depend on prefix length.**
  `smooth_exponential`, `smooth_reverse_exponential`, `smooth_uniform`
  are O(n). `smooth_gaussian` and `smooth_time_decay` are O(n^2) because
  their weight shape depends on the prefix and cannot be precomputed
  once. If you add a smoother, prefer the online form; fall back to the
  naive `for k in range(n): weighted_mean(vectors[:k+1], weights(k+1))`
  only when the weights genuinely vary with prefix length.
