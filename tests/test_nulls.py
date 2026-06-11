"""Null models and the turning-cosine lemma (package invariant)."""
import json
from pathlib import Path

import numpy as np
import pytest

import embflow as ef


@pytest.fixture
def vectors():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((20, 16))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def labels():
    return np.array(["user", "assistant"] * 10)


class TestShuffle:
    def test_preserves_multiset(self, vectors):
        rng = np.random.default_rng(0)
        out = ef.shuffle(vectors, rng)
        np.testing.assert_allclose(
            np.sort(out, axis=0), np.sort(vectors, axis=0), atol=1e-12
        )

    def test_returns_copy(self, vectors):
        before = vectors.copy()
        ef.shuffle(vectors, np.random.default_rng(0))
        np.testing.assert_allclose(vectors, before)

    def test_actually_permutes(self, vectors):
        out = ef.shuffle(vectors, np.random.default_rng(0))
        assert not np.allclose(out, vectors)


class TestRoleSlotShuffle:
    def test_permutes_within_label_groups_only(self, vectors, labels):
        rng = np.random.default_rng(1)
        out = ef.role_slot_shuffle(vectors, labels, rng)
        for r in ("user", "assistant"):
            idx = np.where(labels == r)[0]
            np.testing.assert_allclose(
                np.sort(out[idx], axis=0), np.sort(vectors[idx], axis=0), atol=1e-12
            )

    def test_preserves_mean_embedding(self, vectors, labels):
        out = ef.role_slot_shuffle(vectors, labels, np.random.default_rng(2))
        np.testing.assert_allclose(out.mean(axis=0), vectors.mean(axis=0), atol=1e-12)

    def test_matches_motionlib_fixture(self):
        fix = json.loads(
            (Path(__file__).parent / "fixtures" / "motionlib_fixture.json").read_text()
        )
        rng = np.random.default_rng(fix["seed"])
        E = rng.standard_normal((fix["n"], fix["d"]))
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        labels = np.array(["user", "assistant"] * (fix["n"] // 2))
        out = ef.role_slot_shuffle(E, labels, np.random.default_rng(0))
        np.testing.assert_allclose(
            out[:2], fix["role_slot_shuffle_seed0_first2"], atol=1e-9
        )


class TestTurningCosineLemma:
    """Package invariant: E[turning cosine] = -1/2 for exchangeable unit
    vectors, EXACTLY, independent of anisotropy (E<v1,v2> = mu - 1 and
    E||v||^2 = 2 - 2mu). Observed -0.493 on 1,269 real conversations'
    shuffled user sequences."""

    def test_iid_unit_vectors(self):
        rng = np.random.default_rng(11)
        cosines = []
        for _ in range(300):
            E = rng.standard_normal((30, 16))
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            t = ef.turning_cosines(E)
            cosines.extend(t[~np.isnan(t)])
        assert abs(np.mean(cosines) - (-0.5)) < 0.02

    def test_independent_of_anisotropy(self):
        """Strong common component (mu >> 0): mean turning cosine still -1/2."""
        rng = np.random.default_rng(12)
        base = np.zeros(16)
        base[0] = 1.0
        cosines = []
        for _ in range(300):
            E = base + 0.5 * rng.standard_normal((30, 16))
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            t = ef.turning_cosines(E)
            cosines.extend(t[~np.isnan(t)])
        assert abs(np.mean(cosines) - (-0.5)) < 0.03


class TestNullCorrected:
    def test_scalar_stat(self, vectors):
        real, null_mean, diff = ef.null_corrected(
            lambda E: ef.tortuosity(E), vectors, K=5, seed=42
        )
        assert real == pytest.approx(ef.tortuosity(vectors))
        assert diff == pytest.approx(real - null_mean)

    def test_dict_stat(self, vectors):
        real, null_mean, diff = ef.null_corrected(
            lambda E: ef.motion_signature(E, with_alpha=False), vectors, K=3, seed=0
        )
        assert set(real) == set(null_mean) == set(diff)
        for k in real:
            assert diff[k] == pytest.approx(real[k] - null_mean[k], nan_ok=True)

    def test_labels_use_role_slot_shuffle(self, vectors, labels):
        """A stat reading only the label-position multiset is invariant
        under role-slot shuffle, so its order effect must be ~0."""
        user_positions = labels == "user"
        stat = lambda E: float(np.linalg.norm(E[user_positions].mean(axis=0)))
        real, null_mean, diff = ef.null_corrected(
            stat, vectors, labels=labels, K=4, seed=1
        )
        assert abs(diff) < 1e-9

    def test_full_shuffle_breaks_label_stat(self, vectors, labels):
        """The same stat under FULL shuffle is not preserved (sanity check
        that labels= actually switches the null)."""
        user_positions = labels == "user"
        stat = lambda E: float(np.linalg.norm(E[user_positions].mean(axis=0)))
        real, null_mean, diff = ef.null_corrected(stat, vectors, K=4, seed=1)
        assert abs(diff) > 1e-6

    def test_deterministic_under_seed(self, vectors):
        a = ef.null_corrected(lambda E: ef.tortuosity(E), vectors, K=3, seed=7)
        b = ef.null_corrected(lambda E: ef.tortuosity(E), vectors, K=3, seed=7)
        assert a == b


class TestPairedStats:
    def test_hand_computed(self):
        real = np.array([1.0, 2.0, 3.0, np.nan])
        shuf = np.array([0.5, 1.0, 2.5, 1.0])
        out = ef.paired_stats(real, shuf)
        assert out["n"] == 3
        assert out["mean_real"] == pytest.approx(2.0)
        assert out["mean_shuffled"] == pytest.approx(4.0 / 3)
        assert out["mean_diff"] == pytest.approx(2.0 / 3)
        assert out["frac_positive_diff"] == pytest.approx(1.0)
        d = (2.0 / 3) / (np.std([0.5, 1.0, 0.5]) + 1e-8)
        assert out["cohens_d_paired"] == pytest.approx(d)
