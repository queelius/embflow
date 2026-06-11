"""Backend tests: cache + protocol. No network access anywhere."""
import numpy as np
import pytest

import embflow as ef
from embflow.backends import _require


def make_fake_embed(d=8):
    """Deterministic per-text embedder that counts calls."""
    calls = {"n": 0, "texts": []}

    def embed(texts):
        calls["n"] += 1
        calls["texts"].extend(texts)
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.standard_normal(d))
        return np.asarray(out, dtype=np.float32)

    return embed, calls


class TestCachedEmbedFn:
    def test_roundtrip_values(self, tmp_path):
        embed, _ = make_fake_embed()
        cached = ef.cached_embed_fn(embed, tmp_path / "c.sqlite", "test|8")
        texts = ["alpha", "beta", "gamma"]
        np.testing.assert_allclose(cached(texts), embed(texts), atol=1e-6)

    def test_second_call_hits_cache(self, tmp_path):
        embed, calls = make_fake_embed()
        cached = ef.cached_embed_fn(embed, tmp_path / "c.sqlite", "test|8")
        cached(["a", "b"])
        n_after_first = calls["n"]
        out = cached(["a", "b"])
        assert calls["n"] == n_after_first  # no new underlying calls
        assert out.shape == (2, 8)

    def test_partial_miss_only_embeds_missing(self, tmp_path):
        embed, calls = make_fake_embed()
        cached = ef.cached_embed_fn(embed, tmp_path / "c.sqlite", "test|8")
        cached(["a", "b"])
        calls["texts"].clear()
        cached(["a", "b", "c"])
        assert calls["texts"] == ["c"]

    def test_namespaces_isolate(self, tmp_path):
        embed1, _ = make_fake_embed()
        embed2 = lambda texts: np.ones((len(texts), 8), dtype=np.float32)
        c1 = ef.cached_embed_fn(embed1, tmp_path / "c.sqlite", "model-A")
        c2 = ef.cached_embed_fn(embed2, tmp_path / "c.sqlite", "model-B")
        a = c1(["x"])
        b = c2(["x"])
        assert not np.allclose(a, b)

    def test_empty_input(self, tmp_path):
        embed, calls = make_fake_embed()
        cached = ef.cached_embed_fn(embed, tmp_path / "c.sqlite", "t")
        out = cached([])
        assert out.shape[0] == 0
        assert calls["n"] == 0

    def test_persists_across_instances(self, tmp_path):
        embed, calls = make_fake_embed()
        path = tmp_path / "c.sqlite"
        ef.cached_embed_fn(embed, path, "t")(["x"])
        embed2, calls2 = make_fake_embed()
        out = ef.cached_embed_fn(embed2, path, "t")(["x"])
        assert calls2["n"] == 0
        assert out.shape == (1, 8)


class TestRequire:
    def test_missing_module_raises_with_hint(self):
        with pytest.raises(RuntimeError, match="pip install some-package"):
            _require("definitely_not_a_real_module_xyz", "some-package")


class TestProviderFactories:
    def test_openai_requires_openai(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name == "openai":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(RuntimeError, match="pip install openai"):
            ef.openai_embed_fn()

    def test_ollama_requires_ollama(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def block(name, *a, **k):
            if name == "ollama":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", block)
        with pytest.raises(RuntimeError, match="pip install ollama"):
            ef.ollama_embed_fn()
