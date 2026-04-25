# Experiments

Self-contained experiments using `embflow`'s primitives. Each experiment
is a subpackage with a runnable entry point:

```bash
python -m experiments.<name>.run
```

## Catalog

### `prefix_vs_smoothed/`

Tests whether the prefix-embedding path
$t_k = \mathrm{embed}(x_1 \cdots x_k)$
can be approximated by a weighted combination of per-message embeddings
$t_k^{(w)} = \mathrm{normalize}\bigl(\sum_{j \le k} w_{kj} \cdot \mathrm{embed}(x_j)\bigr)$
under several weighting schemes (uniform, length-weighted, exponential
sweep, per-conversation least-squares ceiling).

Why: prefix embeddings are state-like and well-behaved under embflow's
geometric operators, but cost O(k) per step (re-encode every prefix).
Per-item embeddings cache cheaply, but their raw path is jumpy. If a
fixed weighting reproduces the prefix path, we have a cheap drop-in
approximation; how good the approximation is is an empirical question
that depends on how contextual the embedder is.

See `experiments/prefix_vs_smoothed/README.md`.

## Running

```bash
# Smoke test: synthetic corpus + TF-IDF, runs out of the box.
python -m experiments.prefix_vs_smoothed.run

# Real corpus + sentence-transformers
pip install sentence-transformers
python -m experiments.prefix_vs_smoothed.run \
    --embedder sentence-transformers \
    --corpus path/to/conversations.jsonl \
    --n 100

# OpenAI embeddings
pip install openai
export OPENAI_API_KEY=sk-...
python -m experiments.prefix_vs_smoothed.run \
    --embedder openai \
    --corpus path/to/conversations.jsonl \
    --n 100
```

Outputs land in `experiments/<name>/results/` as `summary.json` (per-
candidate aggregates) and `raw.json` (per-conversation runs). The
`results/` directory is generated; treat it as build output.

## Adding an experiment

Each subpackage should expose:

- `corpus.py`: how to load input data
- `embedders.py` or equivalent: pluggable models
- `metrics.py`: comparison metrics
- `run.py`: orchestration with a `__main__` entry point
- `README.md`: motivation, design, interpretation

Experiments depend on `embflow` only; if an experiment needs a model
provider (HuggingFace, OpenAI), import lazily and fail with an
informative message when the dependency is absent.
