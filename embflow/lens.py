"""Lens: composable weight functions over embedding sequences.

A Lens produces a weight vector for a sequence of embeddings.
The weight can depend on position, metadata, and the vectors themselves.

Lenses compose via multiplication (*): the weights multiply pointwise.
Uniform() is the identity element.

Each lens defines weights(vectors, meta, j) -> ndarray of shape (j+1,),
giving the weight for items 0..j when projecting at position j.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Lens:
    """Base class for embedding sequence lenses."""

    def weights(self, vectors, meta, j):
        """Return weight vector of shape (j+1,) for projection at position j.

        Parameters
        ----------
        vectors : ndarray of shape (n, d)
        meta : list of dicts, length n (optional, can be None)
        j : int, the endpoint index (0 <= j < n)

        Returns
        -------
        ndarray of shape (j+1,)
        """
        raise NotImplementedError

    def project(self, vectors, meta=None):
        """Fold: weighted average of the full sequence.

        Returns normalized embedding vector of shape (d,).
        """
        n = len(vectors)
        w = self.weights(vectors, meta, n - 1)
        return _normalize(np.average(vectors[:n], axis=0, weights=w))

    def trajectory(self, vectors, meta=None):
        """Scan: running projection at each position.

        Returns ndarray of shape (n, d), where result[j] is the
        projection of vectors[0:j+1].
        """
        n = len(vectors)
        result = np.empty_like(vectors)
        for j in range(n):
            w = self.weights(vectors, meta, j)
            result[j] = _normalize(np.average(vectors[:j + 1], axis=0, weights=w))
        return result

    def __mul__(self, other):
        """Compose lenses by multiplying weights pointwise."""
        return _ComposedLens(self, other)

    def __rmul__(self, other):
        if isinstance(other, Lens):
            return _ComposedLens(other, self)
        return NotImplemented

    def half_life(self):
        """Semantic half-life in items, if applicable. None otherwise."""
        return None

    def describe(self):
        """Human-readable description of this lens."""
        return self.__class__.__name__


class _ComposedLens(Lens):
    """Product of two lenses (weights multiply pointwise)."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def weights(self, vectors, meta, j):
        wa = self.a.weights(vectors, meta, j)
        wb = self.b.weights(vectors, meta, j)
        return wa * wb

    def describe(self):
        return f"({self.a.describe()} * {self.b.describe()})"


class Uniform(Lens):
    """Equal weight for all items. Identity element for composition."""

    def weights(self, vectors, meta, j):
        return np.ones(j + 1)

    def describe(self):
        return "Uniform()"


class Exponential(Lens):
    """Exponential recency weighting.

    w_k = alpha^(j - k) for k = 0..j.
    Recent items get more weight. alpha=1 degenerates to Uniform.

    Parameters
    ----------
    alpha : float in (0, 1]
        Decay rate. Lower = shorter memory.
    """

    def __init__(self, alpha=0.85):
        self.alpha = alpha

    def weights(self, vectors, meta, j):
        return np.array([self.alpha ** (j - k) for k in range(j + 1)])

    def half_life(self):
        if self.alpha <= 0 or self.alpha >= 1:
            return float("inf")
        return np.log(0.5) / np.log(self.alpha)

    def describe(self):
        hl = self.half_life()
        return f"Exponential(alpha={self.alpha}, half_life={hl:.1f})"


class ReverseExponential(Lens):
    """Exponential primacy weighting.

    w_k = alpha^k for k = 0..j.
    Earlier items get more weight.
    """

    def __init__(self, alpha=0.85):
        self.alpha = alpha

    def weights(self, vectors, meta, j):
        return np.array([self.alpha ** k for k in range(j + 1)])

    def describe(self):
        return f"ReverseExponential(alpha={self.alpha})"


class Gaussian(Lens):
    """Gaussian focal weighting.

    Peaked at a focal point. Items near the focus get highest weight.

    Parameters
    ----------
    focus : float
        Focal point as fraction of sequence length (0.0 = start, 1.0 = end).
        Or an int for absolute position.
    sigma : float or None
        Standard deviation. None = auto (1/6 of sequence length).
    """

    def __init__(self, focus=0.5, sigma=None):
        self.focus = focus
        self.sigma = sigma

    def weights(self, vectors, meta, j):
        n = j + 1
        if isinstance(self.focus, float) and self.focus <= 1.0:
            k0 = self.focus * (n - 1)
        else:
            k0 = self.focus
        sigma = self.sigma if self.sigma is not None else max(n / 6, 1)
        return np.array([np.exp(-((k - k0) ** 2) / (2 * sigma ** 2)) for k in range(n)])

    def describe(self):
        return f"Gaussian(focus={self.focus}, sigma={self.sigma})"


class Surprise(Lens):
    """Surprise weighting: items that deviate from the running mean.

    w_k = 1 - cos(e_k, running_mean[0:k]).
    Higher weight for messages that shift the topic.
    """

    def weights(self, vectors, meta, j):
        n = j + 1
        w = np.ones(n)
        running_sum = np.zeros(vectors.shape[1])
        for k in range(n):
            if k > 0:
                running_mean = _normalize(running_sum / k)
                sim = float(_cosine_sim(
                    vectors[k].reshape(1, -1),
                    running_mean.reshape(1, -1)
                )[0, 0])
                w[k] = max(1 - sim, 0.01)
            running_sum += vectors[k]
        return w

    def describe(self):
        return "Surprise()"


class FieldWeight(Lens):
    """Weight by a categorical or numeric metadata field.

    For categorical: provide a mapping {value: weight}.
    Unmapped values get weight 1.0.

    For numeric: use the field value directly as weight
    (pass no mapping).

    Parameters
    ----------
    field : str
        Metadata field name.
    mapping : dict or None
        {field_value: weight}. If None, use numeric field value directly.
    default : float
        Weight for unmapped categorical values.
    """

    def __init__(self, field, mapping=None, default=1.0):
        self.field = field
        self.mapping = mapping
        self.default = default

    def weights(self, vectors, meta, j):
        if meta is None:
            return np.ones(j + 1)
        n = j + 1
        w = np.ones(n)
        for k in range(n):
            val = meta[k].get(self.field) if k < len(meta) else None
            if self.mapping is not None:
                w[k] = self.mapping.get(val, self.default)
            elif val is not None:
                w[k] = float(val)
        return w

    def describe(self):
        if self.mapping:
            return f"FieldWeight({self.field!r}, {self.mapping})"
        return f"FieldWeight({self.field!r})"


class TimeDecay(Lens):
    """Exponential decay by real timestamp.

    w_k = exp(-(t_last - t_k) / tau)

    Unlike Exponential (which decays by index), this decays by
    actual elapsed time. Requires a 'timestamp' field in metadata
    (as datetime or epoch float).

    Parameters
    ----------
    half_life_seconds : float
        Time for weight to halve.
    time_field : str
        Metadata field containing timestamp.
    """

    def __init__(self, half_life_seconds=3600, time_field="timestamp"):
        self.half_life_seconds = half_life_seconds
        self.time_field = time_field
        self._tau = half_life_seconds / np.log(2)

    def weights(self, vectors, meta, j):
        if meta is None:
            return np.ones(j + 1)
        n = j + 1
        times = []
        for k in range(n):
            t = meta[k].get(self.time_field) if k < len(meta) else None
            if t is None:
                times.append(0.0)
            elif isinstance(t, (int, float)):
                times.append(float(t))
            else:
                # Assume datetime-like
                times.append(t.timestamp() if hasattr(t, "timestamp") else 0.0)
        t_last = times[j]
        return np.array([np.exp(-(t_last - t) / self._tau) if t <= t_last else 1.0
                         for t in times[:n]])

    def half_life(self):
        return self.half_life_seconds

    def describe(self):
        return f"TimeDecay(half_life={self.half_life_seconds}s)"


class Custom(Lens):
    """Arbitrary weight function.

    Parameters
    ----------
    weight_fn : callable
        (vectors, meta, j) -> ndarray of shape (j+1,)
    name : str
        Description for repr.
    """

    def __init__(self, weight_fn, name="Custom"):
        self._fn = weight_fn
        self._name = name

    def weights(self, vectors, meta, j):
        return self._fn(vectors, meta, j)

    def describe(self):
        return f"Custom({self._name})"
