# embflow

A calculus over embedding sequences. Lenses, trajectories, derivatives, segmentation, and comparison.

## Install

```bash
pip install embflow
```

## Quick Start

```python
import numpy as np
import embflow as ef

# A sequence of embedding vectors (from any source)
vectors = np.random.randn(50, 256)

# Project through a lens (fold to single embedding)
emb = ef.Exponential(0.85).project(vectors)

# Compute the trajectory (scan: running projection at each step)
traj = ef.Exponential(0.85).trajectory(vectors)

# Derivatives: how is the trajectory changing?
vel = ef.velocity(traj)     # first derivative (transition vectors)
curv = ef.curvature(traj)   # second derivative (turning points)

# Compose lenses (weights multiply pointwise)
lens = ef.Exponential(0.85) * ef.FieldWeight("role", {"user": 3.0})
emb = lens.project(vectors, meta)

# Segment at changepoints
segments = ef.auto_segment(vectors, alpha=0.85)

# Compare trajectories
dist = ef.trajectory_distance(traj_a, traj_b, method="dtw")
```
