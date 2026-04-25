"""Build trajectories from a conversation: prefix and per-item.

Path A (prefix path, "ground truth"):
    t_k = embed(x_1 || ... || x_k)
where || is string concatenation. This is what we want to approximate.

Path B (per-item embeddings):
    e_k = embed(x_k)
The candidate trajectories are weighted combinations of these.
"""
import numpy as np


def normalize_rows(matrix):
    """Unit-normalize each row; rows with zero norm pass through unchanged."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=matrix.copy(), where=norms > 0)


def prefix_path(messages, embedder, joiner=" "):
    """Embed the running concatenation of messages up to each k.

    Returns an (n, d) ndarray where row k is the embedding of
    ``joiner.join(messages[0:k+1])``.

    NOTE: this is O(n) embedder calls, each on increasingly long input.
    Fine for the experiment (short conversations); not a production path.
    """
    accum = []
    prefixes = []
    for m in messages:
        accum.append(m)
        prefixes.append(joiner.join(accum))
    return embedder(prefixes)


def per_item_embeddings(messages, embedder):
    """Embed each message independently. Returns (n, d) ndarray."""
    return embedder(list(messages))
