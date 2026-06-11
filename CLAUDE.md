# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`embflow` is a small Python library (~2000 LOC, alpha) that treats an ordered
sequence of embedding vectors as a path in R^d and provides a calculus over
such paths: weighted folds, scan-style smoothing, differential operators,
second-order geometry, motion statistics, null models, segmentation,
distance, and a validation protocol. It is embedding-source-agnostic: any
producer of vectors is upstream, embflow is downstream. `numpy` and
`scikit-learn` (only for `cosine_similarity`) are the sole runtime deps;
`openai` and `ollama` are optional lazy deps in `backends.py`. Python 3.10+.

The v0.2.0 rewrite removed the `Lens` class hierarchy and replaced it with
plain functions (weight generators + smoothers). The organizing concept is
now the trajectory itself, not the projection onto it. The v0.3.0 release
aligned the package with the embedding-dynamics paper (the paper cites
embflow as the reference implementation of the calculus): paper conventions
are non-negotiable and ported code is pinned to the paper's `motionlib` by
fixture (see Design conventions).

**Lens convention (package-wide, paper-aligned):** `w(j,k) = alpha^(k-j)`.
HIGHER alpha = LONGER memory; alpha -> 1 approaches the running mean;
half-life `h = log(0.5)/log(alpha)` steps. Document this at every
alpha-taking function. The dynamics are linear (`s_k = alpha*s_{k-1} +
m_k*e_k`, the Euler step of `dx/dt = -lambda*x + f(t)` with
`alpha = e^(-lambda dt)`); unit normalization is a READOUT for cosine
comparison, not part of the dynamics.

## Commands

```bash
pip install -e ".[dev]"                    # editable install with pytest + pytest-cov
pytest                                     # full test suite (tests/test_*.py, split per module)
pytest tests/test_core.py::TestWeightComposition::test_multiplication_is_commutative  # one test
pytest --cov=embflow --cov-report=term-missing                                         # coverage
python -m build                            # build sdist/wheel (version from embflow.__version__)
python -m experiments.prefix_vs_smoothed.run                                           # smoke test the experiment scaffold
python tests/fixtures/generate_motionlib_fixture.py    # regenerate motionlib fidelity fixture
```

`pyproject.toml` pulls `version` dynamically from `embflow/__init__.py`, so
bump `__version__` there when cutting a release, not in `pyproject.toml`.

## Architecture

Eight source modules organized around a clear axis. All public names are
re-exported from `embflow/__init__.py`; add new symbols there or external
callers will not see them.

### `state.py`: linear dynamics vs normalized readout

The conceptual core of 0.3.0. `leaky_state(E, alpha, masses=None)` returns
the raw `(n, d)` states `s_k = alpha*s_{k-1} + m_k*e_k`; `trajectory(...)`
is its unit-row readout. `trajectory(E, alpha)` agrees with
`smooth_exponential(E, alpha)` to float precision (the weighted-mean
denominator is a positive scalar and cannot change direction), so the two
coexist; use `trajectory` when you need masses or raw states.
`alpha_to_half_life` / `half_life_to_alpha` convert between the two
parameterizations of memory length.

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
`angular_velocity` (1 - cos between adjacent trajectory POINTS),
`turning_cosines` (cos between consecutive VELOCITIES, NaN at duplicate
points). The latter two are deliberately distinct; do not conflate them.

**Motion scalars (motionlib ports)**: `tortuosity` (windowed
net-displacement / path-length), `speed_autocorr` (lag-k autocorrelation
of the speed series; NOT a pure order statistic, report it
null-corrected), `motion_signature` (the per-sequence "gait" dict:
speed mean/std, turning mean/std, speed_ac1, tortuosity, alpha_hat).

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

**Adaptive analysis**: `adaptive_alpha(vectors, grid=None, max_messages=400)`
is the motionlib port (vectorized over the grid): argmax of
`mean_k cos(x_{k-1}(alpha), e_k)`. Default grid `ALPHA_GRID` =
{0.05..0.95 step 0.05} plus 0.999; NaN for n < 3; capped at 400 messages.
Near-ties tilt toward LONGER memory because the EPS-guarded normalization
scores `norm/(norm+EPS)` and higher alpha accumulates a larger-norm state
(a constant sequence fits 0.999, identical to motionlib). This EPS
placement is load-bearing; do not "clean it up".

`structural_richness(vectors, weight_fns=None)` computes the mean pairwise
cosine distance over a set of weighted projections. The default set is
`[uniform, exponential(0.85), reverse_exponential(0.85), novelty]`. Users
can pass a list of `(vectors, meta) -> (n,)` callables for custom
viewpoints.

Shared `_auto_threshold(signal, k=1.5)` helper is used by both `peaks()`
and `_recursive_changepoints`. The latter uses `k=1.0` deliberately: it's
a split-gate (any peak above threshold allows a split, then the strongest
is chosen), so a laxer multiplier is correct there.

### `nulls.py`: order vs composition

`shuffle(E, rng)` (exchangeable null), `role_slot_shuffle(E, labels, rng)`
(permutes within label groups; preserves role pattern and per-label
multiset), `null_corrected(stat_fn, E, labels=None, K=5, seed=42)` ->
`(real, null_mean, diff)` for float- or dict-valued stats, and
`paired_stats(real, shuffled)` (cohort-level paired Cohen's d). The
turning-cosine lemma is a package invariant pinned by tests: for
exchangeable unit vectors, E[turning cosine] = -1/2 EXACTLY, independent
of anisotropy. Any motion statistic that survives shuffling is secretly
measuring content.

### `validate.py`: the prefix-validation protocol

`prefix_experiment(conversations, embed_fn, ...)` is the embedding-dynamics
paper's gating experiment (exp04) as a library protocol: lens family
(`default_lenses()`: uniform/exp/primacy, each with and without token-mass
weighting), agreement curves per prefix, mismatched-pair and cross-prefix
nulls, pooled top-1 retrieval, alpha coherence, and the gate (best-lens
mean >= 0.8 and >= null + 0.25). Requires >= 2 conversations with >= 2
valid prefixes. `lens_weights(kind, param, k)` is the weight primitive.

### `backends.py`: embedding sources

The `embed_fn` protocol (`(list[str]) -> (n, d) ndarray`) is the only
contract `validate` relies on. `cached_embed_fn(fn, path, namespace)`
wraps any embed_fn with a sqlite content-hash cache (namespace MUST encode
model identity). `openai_embed_fn` / `ollama_embed_fn` import their
providers lazily and raise RuntimeError with pip hints when missing,
keeping core embflow numpy-only. Providers cache RAW vectors and
normalize on the way out.

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

`velocity_gram(traj)` returns the `(n-1, n-1)` Gram matrix of velocities:
rotation+translation-invariant internal geometry (the RSA/CKA comparison
level). Scope note from the 2026-06-10 experiments: LOCAL cross-topic
motion motifs tested NEGATIVE at scalar-channel representation;
sequence-level signatures and arc-scale structure are where the signal is.

`continuation_score(a, b, alpha)` uses the new weight primitives:
`weighted_mean(a, exponential_weights(len(a), alpha))` (recency end of A)
compared to `weighted_mean(b, reverse_exponential_weights(len(b), alpha))`
(primacy start of B). "Does B pick up where A left off."

## Experiments

Self-contained research scripts live under `experiments/`. Each is a
subpackage with its own README and a runnable entry point:

```bash
python -m experiments.<name>.run
```

Currently:

- `prefix_vs_smoothed/`: tests whether weighted combinations of
  per-message embeddings can approximate the prefix path
  `embed(x_1..x_k)`. Pluggable corpora (synthetic generator with known
  changepoints, plus a JSONL loader), embedders (TF-IDF +
  sentence-transformers + OpenAI adapters with lazy imports), weighting
  schemes (uniform, length-weighted, exponential sweep, per-conversation
  least-squares ceiling), and metrics (pointwise cosine, DTW, shape,
  operator correlations, segmentation Jaccard, ground-truth changepoint
  recall). Runs out of the box on the synthetic corpus + TF-IDF; no
  extra deps required.

Conventions:

- Experiments depend on `embflow` only. Optional model providers
  (sentence-transformers, openai) import lazily and raise with install
  hints when missing.
- Outputs land in `experiments/<name>/results/` as `summary.json`
  (per-candidate aggregates) and `raw.json` (per-conversation runs).
  The `results/` directory is gitignored; treat it as build output.
- Add a new experiment as `experiments/<name>/` with at minimum
  `__init__.py`, `run.py` (with a `__main__` block), and a `README.md`
  documenting motivation and how to interpret outputs.

## Design conventions

- **The lens convention is documented at every alpha-taking function.**
  w(j,k) = alpha^(k-j); HIGHER alpha = LONGER memory; half-life
  log(0.5)/log(alpha). An earlier paper draft had smoothing labels
  inverted because this was ambiguous; never leave an alpha undocumented.
- **motionlib fidelity is pinned by fixture.**
  `tests/fixtures/motionlib_fixture.json` stores the authoritative
  outputs of the paper's reference implementation
  (`~/github/cmri/papers/embedding-dynamics/experiments/motionlib.py`)
  on a seed-regenerated input; `tests/test_motion.py` and
  `tests/test_nulls.py` compare embflow's ports against it. Regenerate
  with `python tests/fixtures/generate_motionlib_fixture.py` only when
  motionlib itself changes.
- **Tests are split per module**: `tests/test_core.py` (weights,
  smoothers, operators, comparison, public API), `test_state.py`,
  `test_motion.py`, `test_nulls.py`, `test_validate.py`,
  `test_backends.py` (no network anywhere).
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
