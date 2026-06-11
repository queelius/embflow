"""Tests for the prefix-validation protocol (exp04 port)."""
import numpy as np
import pytest

import embflow as ef
from embflow.validate import accumulate, _approx_token_len, _default_prefix


def hash_vec(text, d=32):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def linear_embed(texts):
    """Prefix embedding == token-mass-weighted sum of part embeddings.

    Splits on the prefix joiner; single messages are their own part. With
    this embedder the uniform+mass lens reproduces the prefix path EXACTLY,
    so the experiment must select it and pass the gate.
    """
    out = []
    for t in texts:
        parts = t.split("\n\n")
        v = sum(_approx_token_len(p) * hash_vec(p) for p in parts)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


def make_conversations(n_convs=3, n_msgs=8, seed=0):
    rng = np.random.default_rng(seed)
    convs = []
    for c in range(n_convs):
        msgs = []
        for m in range(n_msgs):
            length = int(rng.integers(20, 120))
            word = f"conv{c}msg{m}"
            msgs.append({"content": (word + " ") * (length // (len(word) + 1) + 1)})
        convs.append(msgs)
    return convs


class TestLensWeights:
    def test_uniform(self):
        np.testing.assert_allclose(ef.lens_weights("uniform", 0.0, 4), np.ones(4))

    def test_exp_recency(self):
        """w(j,k) = alpha^(k-j): last item weight 1, geometric decay backwards."""
        np.testing.assert_allclose(
            ef.lens_weights("exp", 0.5, 4), [0.125, 0.25, 0.5, 1.0]
        )

    def test_primacy(self):
        """w(j) = beta^(j-1): first item weight 1."""
        np.testing.assert_allclose(
            ef.lens_weights("primacy", 0.5, 4), [1.0, 0.5, 0.25, 0.125]
        )

    def test_exp_matches_exponential_weights(self):
        np.testing.assert_allclose(
            ef.lens_weights("exp", 0.85, 7), ef.exponential_weights(7, 0.85)
        )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            ef.lens_weights("nope", 0.5, 3)


class TestDefaultLenses:
    def test_count_and_labels(self):
        lenses = ef.default_lenses()
        assert len(lenses) == 2 * (1 + 9 + 3)
        labels = [l[0] for l in lenses]
        assert "uniform" in labels and "uniform+mass" in labels
        assert "exp0.85" in labels and "primacy0.9+mass" in labels
        assert len(set(labels)) == len(labels)


class TestAccumulate:
    def test_matches_manual_weighted_sum(self):
        rng = np.random.default_rng(2)
        E = rng.standard_normal((6, 8))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        masses = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        k = 4
        w = ef.lens_weights("exp", 0.8, k) * masses[:k]
        expected = (w[:, None] * E[:k]).sum(axis=0)
        expected /= np.linalg.norm(expected)
        got = accumulate(E, masses, "exp", 0.8, True, k)
        np.testing.assert_allclose(got, expected, atol=1e-9)


class TestPrefixExperiment:
    def test_linear_embedder_selects_uniform_mass_and_passes_gate(self):
        convs = make_conversations()
        res = ef.prefix_experiment(convs, linear_embed, n_nulls=50, seed=42)
        assert res["best_lens"] == "uniform+mass"
        assert res["lens_means"]["uniform+mass"] > 0.999
        assert res["gate"]["passed"] is True
        assert res["retrieval_top1_conv"] == 1.0
        assert res["retrieval_top1_exact"] > 0.9

    def test_result_keys(self):
        convs = make_conversations(2, 6)
        res = ef.prefix_experiment(convs, linear_embed, n_nulls=20)
        for key in [
            "lens_means", "best_lens", "null_mismatched_mean",
            "null_crossprefix_mean", "retrieval_top1_exact",
            "retrieval_top1_conv", "alpha_coherence", "gate",
            "per_conversation",
        ]:
            assert key in res, key
        assert set(res["gate"]) == {"best_lens_mean", "criterion", "passed"}
        for entry in res["alpha_coherence"]:
            assert set(entry) == {"id", "best_fit_alpha", "adaptive_alpha"}

    def test_max_prefix_tokens_truncates_valid_k(self):
        convs = make_conversations(2, 8)
        cap = _approx_token_len(_default_prefix(convs[0], 4))
        res = ef.prefix_experiment(
            convs, linear_embed, max_prefix_tokens=cap, n_nulls=20
        )
        cid = list(res["per_conversation"])[0]
        assert res["per_conversation"][cid]["K"] <= 4

    def test_custom_ids(self):
        convs = make_conversations(2, 6)
        res = ef.prefix_experiment(convs, linear_embed, ids=["A", "B"], n_nulls=10)
        assert set(res["per_conversation"]) == {"A", "B"}

    def test_fewer_than_two_conversations_raises(self):
        with pytest.raises(ValueError):
            ef.prefix_experiment(make_conversations(1), linear_embed)

    def test_deterministic_under_seed(self):
        convs = make_conversations(2, 6)
        a = ef.prefix_experiment(convs, linear_embed, n_nulls=30, seed=9)
        b = ef.prefix_experiment(convs, linear_embed, n_nulls=30, seed=9)
        assert a["null_mismatched_mean"] == b["null_mismatched_mean"]

    def test_role_tagged_prefixes(self):
        """Messages with roles use the [role]: content prefix format."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _default_prefix(msgs, 2) == "[user]: hello\n\n[assistant]: hi"


class TestPrefixExperimentEdges:
    def test_mismatched_ids_raise(self):
        convs = make_conversations(2, 6)
        with pytest.raises(ValueError, match="ids"):
            ef.prefix_experiment(convs, linear_embed, ids=["only-one"])

    def test_empty_conversation_raises(self):
        convs = make_conversations(2, 6)
        convs[0] = []
        with pytest.raises(ValueError, match="empty"):
            ef.prefix_experiment(convs, linear_embed)

    def test_alpha_coherence_without_mass_lenses(self):
        """exp lenses without mass variants: the fallback list is used."""
        lenses = [("uniform", "uniform", 0.0, False),
                  ("exp0.8", "exp", 0.8, False)]
        convs = make_conversations(2, 6)
        res = ef.prefix_experiment(convs, linear_embed, lenses=lenses, n_nulls=10)
        for entry in res["alpha_coherence"]:
            assert entry["best_fit_alpha"] == 0.8
