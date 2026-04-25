"""Embedder adapters.

Three classes of embedders covering a range of contextuality:

- ``TfidfEmbedder``  (low):    bag-of-words. Linear; concatenation = sum.
- ``SentenceTransformersEmbedder`` (medium): averaged-token sentence encoder.
- ``OpenAIEmbedder``  (high):   fully contextual transformer pooling.

Only TF-IDF is guaranteed to work out of the box; the other two are
optional and report a clear error if their dependencies are missing.

Each adapter exposes:
    name           : str
    contextuality  : "low" | "medium" | "high"
    __call__(texts) -> ndarray of shape (n, d)
"""
import numpy as np


class TfidfEmbedder:
    """Bag-of-words TF-IDF.

    Must be ``fit()``ed on a corpus before use. Vocabulary is derived
    from that corpus, so out-of-vocabulary tokens at inference time
    simply contribute zero.

    For TF-IDF, ``embed(x_1 + ... + x_k) ~ length-weighted sum of
    embed(x_j)``, so the length-weighted candidate should achieve
    near-perfect equivalence. This is the experiment's positive control.
    """

    name = "tfidf"
    contextuality = "low"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus):
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def __call__(self, texts):
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbedder must be .fit(corpus) before use"
            )
        return self.vectorizer.transform(texts).toarray().astype(float)


class SentenceTransformersEmbedder:
    """Averaged-token sentence encoder (e.g., all-MiniLM-L6-v2).

    Approximate equivalence to the prefix path is expected: tokens are
    averaged within a sentence, and concatenation of sentences re-averages
    the same tokens (modulo position embeddings and pooling normalization).
    """

    name = "sentence-transformers"
    contextuality = "medium"

    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    def __call__(self, texts):
        return np.asarray(self.model.encode(list(texts), convert_to_numpy=True))


class OpenAIEmbedder:
    """OpenAI text embedding model (e.g., text-embedding-3-small).

    Fully contextual: cross-attention means tokens in later messages
    influence the contribution of earlier tokens in the prefix embedding.
    Exact linear equivalence is NOT expected here. The interesting
    question is how much of the variance the best linear weighting
    can still recover.

    Requires the ``openai`` package and a configured API key.
    """

    name = "openai"
    contextuality = "high"

    def __init__(self, model="text-embedding-3-small", batch_size=64):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai not installed. Run: pip install openai"
            ) from e
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def __call__(self, texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.model)
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=float)


def make_embedder(name, **kwargs):
    """Construct an embedder by name."""
    if name == "tfidf":
        return TfidfEmbedder(**kwargs)
    if name == "sentence-transformers":
        return SentenceTransformersEmbedder(**kwargs)
    if name == "openai":
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name!r}")
