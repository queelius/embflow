# embflow

A calculus over embedding sequences. Weight generators, smoothers,
differential operators, and trajectory distance for paths through R^d.

## Install

```bash
pip install embflow
```

## Quick start

```python
import numpy as np
import embflow as ef

# A sequence of embedding vectors from any source.
vectors = np.random.randn(50, 256)

# Fold with a weighted mean.
w = ef.exponential_weights(len(vectors), 0.85)   # recency weights
emb = ef.weighted_mean(vectors, w)

# Scan: running projection at every prefix. O(n).
traj = ef.smooth_exponential(vectors, 0.85)

# Derivatives: treat the trajectory as a path in embedding space.
v = ef.velocity(traj)        # first differences
c = ef.curvature(traj)       # second differences
j = ef.jerk(traj)            # third differences
s = ef.speed(traj)            # |velocity|
a = ef.angular_velocity(traj) # 1 - cos between adjacent points

# Global and second-order geometry.
arc = ef.arc_length(traj)                 # cumulative segment lengths
d = ef.drift(traj)                        # cosine distance first-to-last
R = ef.local_curvature_radius(traj)       # circumradius through each triple
cov = ef.velocity_covariance(traj, window=5)  # local structure tensor

# Compose weights with numpy *.
meta = [{"role": "user"} for _ in vectors]
w = (ef.exponential_weights(len(vectors), 0.85)
     * ef.field_weights(meta, "role", {"user": 3.0}))
emb = ef.weighted_mean(vectors, w)

# Segment at changepoints.
segments = ef.auto_segment(vectors, alpha=0.85)

# Compare two trajectories.
dist = ef.trajectory_distance(traj_a, traj_b, method="dtw")
```
