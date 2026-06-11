# embflow

A calculus over embedding sequences. Weight generators, smoothers,
differential operators, motion statistics, null models, and trajectory
distance for paths through R^d. Reference implementation of the
calculus in the embedding-dynamics paper.

## Install

```bash
pip install git+https://github.com/queelius/embflow.git
```

(Not yet on PyPI.) Runtime deps are numpy and scikit-learn only; the
OpenAI and Ollama embedding backends are optional
(`pip install openai` / `pip install ollama`).

## The lens convention (read this first)

Everything alpha-shaped in embflow uses one convention:

> **w(j, k) = alpha^(k-j). Higher alpha = longer memory.**
> alpha -> 1 approaches the running mean; the half-life is
> log(0.5)/log(alpha) steps (`ef.alpha_to_half_life(0.85)` is about 4.3).

The exponential lens is derived, not chosen: the accumulation
`s_k = alpha*s_{k-1} + m_k*e_k` is the Euler step of the overdamped ODE
`dx/dt = -lambda*x + f(t)` with `alpha = e^(-lambda*dt)`. The state is
linear in R^d; unit normalization `x_k = s_k/||s_k||` is a *readout*
for cosine comparison, not part of the dynamics. Fitted on real
conversations, adaptive alpha lands around 0.78 (ChatGPT) to 0.84
(Claude Code), and the ordering replicates across embedding models.

## Quick start

```python
import numpy as np
import embflow as ef

# A sequence of embedding vectors from any source.
vectors = np.random.randn(50, 256)

# Linear state and normalized readout (token counts as masses).
states = ef.leaky_state(vectors, alpha=0.85)          # raw dynamics
traj = ef.trajectory(vectors, alpha=0.85)             # unit readout
traj = ef.trajectory(vectors, 0.85, masses=np.ones(50))

# Fold with a weighted mean; weights compose with numpy *.
w = ef.exponential_weights(len(vectors), 0.85)
emb = ef.weighted_mean(vectors, w)

# Derivatives and motion statistics.
v = ef.velocity(traj)              # first differences
s = ef.speed(traj)                 # |velocity|
t = ef.turning_cosines(vectors)    # cos between consecutive velocities
sig = ef.motion_signature(vectors) # per-sequence "gait" dict
alpha = ef.adaptive_alpha(vectors) # fitted memory length

# Null-correct order statistics (composition vs order).
real, null, diff = ef.null_corrected(
    lambda E: ef.motion_signature(E, with_alpha=False), vectors
)

# Segment at changepoints; compare trajectories.
segments = ef.auto_segment(vectors, alpha=0.85)
dist = ef.trajectory_distance(traj, traj, method="dtw")
G = ef.velocity_gram(vectors)      # rotation/translation-invariant geometry

# Validate an embedding model against the prefix path.
conversations = [
    [{"role": "user", "content": "embeddings as paths"},
     {"role": "assistant", "content": "trajectories, lenses, motion"}],
    [{"role": "user", "content": "an unrelated topic"},
     {"role": "assistant", "content": "entirely different content"}],
]
emb_fn = ef.openai_embed_fn(cache_path="emb.sqlite")   # or ollama_embed_fn;
result = ef.prefix_experiment(conversations, emb_fn)   # gate + curves
# (openai_embed_fn needs `pip install openai` and OPENAI_API_KEY; any
#  callable (list[str]) -> (n, d) ndarray works as emb_fn.)
```

## Why null models?

Composition alone induces structure in motion statistics: for
exchangeable unit vectors the expected turning cosine is exactly -1/2
(independent of anisotropy), and message eccentricity creates positive
speed autocorrelation even in shuffled sequences. Raw motion statistics
conflate composition and order; `ef.null_corrected` separates them.
Validated order effects on 1,768 real conversations (paired Cohen's d,
real vs shuffled): speed -2.03, tortuosity +1.39, adaptive alpha -1.83.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT (see [LICENSE](LICENSE)). Citation metadata in
[CITATION.cff](CITATION.cff).
