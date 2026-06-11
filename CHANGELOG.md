# Changelog

## 0.3.0 (2026-06-10)

Paper-aligned release: adopts the conventions validated by the
embedding-dynamics experiments (2026-06-10, 1,768 conversations).

### Breaking

- `adaptive_alpha` realigned with the paper's motionlib reference:
  - signature is now `adaptive_alpha(vectors, grid=None, max_messages=400)`
    (was `alpha_range=(0.3, 0.99), step=0.05`);
  - default grid is `ALPHA_GRID`: {0.05..0.95 step 0.05} plus 0.999;
  - returns NaN for sequences shorter than 3 (was 0.85 for n < 5);
  - near-ties resolve toward LONGER memory: the EPS-guarded
    normalization scores norm/(norm+EPS), and higher alpha accumulates
    a larger-norm state. A constant sequence now fits 0.999, the
    running mean, identical to motionlib; previously ties broke toward
    0.85;
  - sequences are capped at 400 messages by default (`max_messages=None`
    to disable).

### Added

- `state.py`: `leaky_state` (linear dynamics s_k = alpha*s_{k-1} + m_k*e_k),
  `trajectory` (normalized readout, with mass support),
  `alpha_to_half_life` / `half_life_to_alpha`. Both state functions
  enforce the documented alpha domain (0, 1] with a ValueError.
- `ops.py`: `turning_cosines` (velocity-angle cosines, distinct from
  `angular_velocity`), `tortuosity`, `speed_autocorr`,
  `motion_signature` (per-sequence gait vector), `ALPHA_GRID`
  (read-only module constant).
- `nulls.py`: `shuffle`, `role_slot_shuffle`, `null_corrected`,
  `paired_stats`. Documents the turning-cosine lemma (expected turning
  cosine is exactly -1/2 for exchangeable unit vectors, independent of
  anisotropy) as a package invariant with a property test.
- `validate.py`: `prefix_experiment`, the paper's gating experiment as
  a library protocol (lens family, mismatched/cross-prefix nulls,
  pooled top-1 retrieval, alpha coherence, gate); `lens_weights`,
  `default_lenses`.
- `backends.py`: `embed_fn` protocol, `cached_embed_fn` (sqlite
  content-hash cache), `openai_embed_fn`, `ollama_embed_fn` (lazy
  imports with install hints; both share one batching/cache/normalize
  pipeline). Wrapped embed_fns are row-count-validated: a misbehaving
  embedder raises a clear ValueError instead of failing downstream
  with shape errors (same check in `prefix_experiment`).
- `compare.py`: `velocity_gram` (rotation/translation-invariant
  internal geometry).
- `smooth_time_decay` accepts `masses=` (mass and time decay compose
  under pointwise `*`).
- Lens convention documented at every alpha-taking function:
  w(j,k) = alpha^(k-j), HIGHER alpha = LONGER memory, half-life
  log(0.5)/log(alpha).
- motionlib fidelity fixture: `tests/fixtures/motionlib_fixture.json`
  pins the ported operators to the paper's reference implementation.

### Fixed

- `python -m build` was broken since 0.2.0 introduced `experiments/`
  (setuptools flat-layout auto-discovery found two top-level packages).
  Package discovery is now explicit (`include = ["embflow*"]`), so
  `experiments/` stays repo-only and never ships.

### Notes

- 0.2.0 removed the `Lens` class API (`ef.Uniform()`, `ef.Exponential(alpha)`,
  `.project(...)`). The only known consumers, the five scripts in the
  semantic-dynamics paper's `experiments/` directory, were not migrated
  and remain broken by design (no shims); they will be updated when that
  paper is next touched.

## 0.2.0 (2026)

- Removed the `Lens` class hierarchy; reorganized around the path in
  embedding space: weight generators + smoothers + operators as plain
  functions.
- Window-based changepoint detection with empirical defaults.

## 0.1.0 (2026)

- Initial release: a calculus over embedding sequences.
