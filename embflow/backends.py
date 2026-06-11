"""Embedding backends: the embed_fn protocol, provider adapters, and a
content-hash sqlite cache.

An ``embed_fn`` is any callable ``(list[str]) -> ndarray of shape (n, d)``
— the only contract the rest of embflow (notably ``validate``) relies
on. Providers here are optional: ``openai`` and ``ollama`` import lazily
and raise with install hints when missing, so embflow's core stays
numpy-only. The sqlite cache keys vectors by a content hash namespaced
by model identity, which keeps repeated experiments reproducible and
cheap (re-running an analysis embeds nothing new).
"""
import hashlib
import sqlite3
from typing import Protocol

import numpy as np

EPS = 1e-8


class EmbedFn(Protocol):
    """Anything that maps a list of texts to an (n, d) ndarray."""

    def __call__(self, texts):
        ...


def _require(module_name, pip_hint):
    """Import an optional dependency or raise with an install hint."""
    try:
        return __import__(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"{module_name} not installed. Run: pip install {pip_hint}"
        ) from e


def _normalized(embed_fn):
    """Wrap an embed_fn so rows come back unit-normalized."""

    def _embed(texts):
        E = np.asarray(embed_fn(texts), dtype=np.float32)
        if E.size == 0:
            return E
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)

    return _embed


def _batched(embed_batch, batch_size):
    """Lift a one-batch embedder (list[str] -> list of vectors) into an
    embed_fn: batching loop, empty-input guard, float32 stacking."""

    def _embed(texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            out.extend(embed_batch(texts[start:start + batch_size]))
        return np.asarray(out, dtype=np.float32)

    return _embed


def _provider(embed_batch, batch_size, cache_path, namespace, normalize):
    """Shared provider wiring: batching, then optional content-hash cache
    (raw vectors), then optional unit-normalized readout."""
    fn = _batched(embed_batch, batch_size)
    if cache_path is not None:
        fn = cached_embed_fn(fn, cache_path, namespace)
    if normalize:
        fn = _normalized(fn)
    return fn


def cached_embed_fn(embed_fn, cache_path, namespace):
    """Wrap any embed_fn with a sqlite content-hash cache.

    Cache key = sha256(f"{namespace}|{text}"). The ``namespace`` MUST
    encode the model identity (name, dimensions, ...) or vectors from
    different models will collide. Vectors are stored as raw float32
    blobs; cache before normalization if you also wrap with a
    normalizer. The connection is per-wrapper and not thread-safe.

    Parameters
    ----------
    embed_fn : EmbedFn
    cache_path : str or Path
        Sqlite file; created (with table) if absent.
    namespace : str
        Model-identity prefix mixed into every key.

    Returns
    -------
    EmbedFn that consults the cache first and embeds only misses.
    Empty input returns a (0, 0) array (the dimension is unknowable
    without embedding something).
    """
    db = sqlite3.connect(str(cache_path))
    db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(text):
        return hashlib.sha256(f"{namespace}|{text}".encode()).hexdigest()

    def _embed(texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = {}
        missing = []
        for i, t in enumerate(texts):
            row = db.execute("SELECT vec FROM emb WHERE key=?", (_key(t),)).fetchone()
            if row:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            vecs = np.asarray(embed_fn([texts[i] for i in missing]), dtype=np.float32)
            if len(vecs) != len(missing):
                raise ValueError(
                    f"embed_fn returned {len(vecs)} rows for {len(missing)} texts"
                )
            for i, v in zip(missing, vecs):
                db.execute(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                    (_key(texts[i]), v.tobytes()),
                )
                out[i] = v
            db.commit()
        return np.stack([out[i] for i in range(len(texts))])

    return _embed


def openai_embed_fn(model="text-embedding-3-small", dimensions=256,
                    batch_size=64, cache_path=None, normalize=True):
    """OpenAI embeddings as an embed_fn. Lazy: requires ``pip install openai``.

    Defaults match the embedding-dynamics paper setup
    (text-embedding-3-small @ 256 dims, unit rows). Pass ``cache_path``
    to memoize by content hash (namespace ``openai|{model}|{dimensions}``;
    raw vectors are cached, normalization applies on read-back). Reads
    the API key from the environment (``OPENAI_API_KEY``).
    """
    openai = _require("openai", "openai")
    client = openai.OpenAI()

    def embed_batch(batch):
        resp = client.embeddings.create(
            model=model, input=batch, dimensions=dimensions
        )
        return [item.embedding for item in resp.data]

    return _provider(embed_batch, batch_size, cache_path,
                     f"openai|{model}|{dimensions}", normalize)


def ollama_embed_fn(model="nomic-embed-text", host=None,
                    batch_size=64, cache_path=None, normalize=True):
    """Ollama embeddings as an embed_fn. Lazy: requires ``pip install ollama``.

    Pass ``cache_path`` to memoize by content hash (namespace
    ``ollama|{model}``; raw vectors are cached, normalization applies on
    read-back). ``host`` overrides the default local server.
    """
    ollama = _require("ollama", "ollama")
    client = ollama.Client(host=host) if host else ollama.Client()

    def embed_batch(batch):
        return client.embed(model=model, input=batch)["embeddings"]

    return _provider(embed_batch, batch_size, cache_path,
                     f"ollama|{model}", normalize)
