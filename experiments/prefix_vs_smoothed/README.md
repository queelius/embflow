# prefix vs. smoothed-per-item

## Motivation

`embflow`'s operators (velocity, curvature, angular velocity, arc length,
trajectory distance, segmentation) all assume the input sequence is a
**state-like trajectory**: adjacent points are close unless something
semantic actually happened between them.

The cleanest producer of such a trajectory is the **prefix path**:
$t_k = \mathrm{embed}(x_1 \cdots x_k)$. Each point is a function of all
prior context, so successive points differ by a small semantic delta.

The cleanest cheap producer is **per-message embeddings**:
$e_k = \mathrm{embed}(x_k)$. These cache forever and are O(1) per new
message, but the path is dominated by per-message lexical/length
variation. With smoothing it can look state-like, but there is no
guarantee that smoothing reproduces the prefix path's information.

This experiment quantifies that gap. For each of several weighting
schemes $w$, we build the candidate trajectory
$t_k^{(w)} = \mathrm{normalize}\bigl(\sum_{j \le k} w_{kj} \cdot e_j\bigr)$
and compare it to the prefix path.

### Predictions

- **TF-IDF (low contextuality)**: length-weighted candidate ≈ prefix
  path almost exactly. Concatenation in TF-IDF space is literally
  vector addition. This is a positive control: if the scaffold is
  correct, this entry should be ~1.0 mean cosine.
- **Averaged-token sentence encoders (medium)**: approximate equivalence
  under length-weighted or exponential weighting; some divergence from
  position embeddings and pooling normalization.
- **Cross-attentional embedders (high)**: irreducible divergence. The
  prefix embedding captures cross-token structure that no linear
  combination of per-message embeddings can recover. The
  per-conversation least-squares ceiling caps how well *any* linear
  combination can do.

### What "good enough" means

Three thresholds, in order of practical importance:

1. **Downstream agreement** (`segmentation_jaccard` and operator
   correlations). If `auto_segment` and the scalar operators agree,
   the candidate is behaviorally interchangeable for the kinds of
   analyses `embflow` is designed for.
2. **Geometric agreement** (`mean_dtw`, `mean_shape_distance`). The
   trajectories trace similar shapes through embedding space.
3. **Pointwise agreement** (`mean_cosine`). The trajectories are close
   at every step. The strictest bar.

A cheap candidate that passes (1) but not (3) is still usable for
production: it's "wrong" pointwise but "right" for what the user does
with it.

## Running

Out of the box (no extra dependencies):

```bash
python -m experiments.prefix_vs_smoothed.run
```

This uses the synthetic corpus (`corpus.synthetic_corpus`) and the TF-IDF
embedder. Results are written to `experiments/prefix_vs_smoothed/results/`.

With a real corpus and a real embedder:

```bash
pip install sentence-transformers
python -m experiments.prefix_vs_smoothed.run \
    --embedder sentence-transformers \
    --corpus path/to/conversations.jsonl \
    --n 100
```

The corpus format is JSONL with one
`{"messages": ["text", "text", ...]}` per line; ShareGPT-style
`[{"role": "user", "content": "..."}, ...]` is also accepted.

## Output

Two files in `results/`:

- `summary.json`: one entry per candidate weighting with aggregated
  metrics across the corpus.
- `raw.json`: per-conversation, per-candidate metric records. Useful
  for plotting distributions or filtering by length.

The console output prints a metrics table plus a "best-by" line for
pointwise cosine and segmentation agreement. On the synthetic corpus,
ground-truth changepoint recall is also reported for the prefix path
and every candidate, since the synthetic generator knows where the
boundaries are by construction.

## What the synthetic-corpus + TF-IDF run reveals

The default smoke test uses TF-IDF on the synthetic corpus. Initial
results worth knowing about before you run on real data:

- **Length-weighted is *not* better than uniform on TF-IDF.** Pure
  prediction was that TF-IDF + length-weighting reproduces the prefix
  exactly; in practice we see ~0.995 vs ~0.997 (uniform slightly ahead).
  The reason is `TfidfVectorizer.transform`'s per-document term-frequency
  normalization: a longer concatenated document does not produce a
  strictly-additive vector, so the "exact linear" relationship is only
  approximate. Real noise floor is around 0.005 mean cosine error.
- **Per-conversation least-squares ceiling = 1.000.** Confirms that the
  prefix path lies in the column span of the per-message embeddings, as
  expected for a linear embedder. The interesting question is whether
  this holds for non-linear embedders.
- **`auto_segment` is mostly silent on the synthetic corpus.** Even at
  `--min-segment-size 2`, only ~10% of conversations have any detected
  changepoint, and ground-truth recall stays under 5%. Two reads of this:
  (1) TF-IDF prefix embeddings smooth too much for short conversations,
  or (2) `auto_segment`'s default thresholds are tuned for longer paths.
  Either way, segmentation metrics on this corpus + embedder combination
  are not informative; rely on operator correlations and pointwise
  cosines for now, and revisit segmentation when running with
  sentence-transformers or OpenAI on longer real conversations.

These findings are *the experiment doing its job*: surfacing where the
cheap approximation works (high pointwise cosine) and where the
geometry-first downstream tests are not yet probing real differences
(segmentation on short, low-contextuality data).

## Caveats

- DTW is O(n × m). The synthetic corpus stays under ~20 messages per
  conversation; for real data with 50+ messages, expect long runtimes.
  Replace DTW with `method="resample"` if it dominates wall time.
- The per-conversation least-squares ceiling is overfit by design: it's
  the answer to "what's the best a linear weighting *could* achieve on
  this exact conversation?" It is not a generalizable model. Use it
  to bound expectations for the parameterized candidates, not as a
  weighting scheme to ship.
- `length_weighted` uses character length. For tokenized embedders,
  token count would be more principled; swap in `len(tokenizer(m))`
  if it matters for your embedder.
- `meta`-aware weightings (role-based, time-based) are deliberately
  out of scope here; this experiment isolates the
  "how well does any linear combination of per-message embeddings
  match the prefix path?" question.
