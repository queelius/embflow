"""Tests for embflow.state: linear dynamics, normalized readout, conventions."""
import numpy as np
import pytest

import embflow as ef


@pytest.fixture
def vectors():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((12, 16))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class TestLeakyState:
    def test_recurrence(self, vectors):
        """s_k = alpha * s_{k-1} + m_k * e_k, s_{-1} = 0."""
        alpha = 0.85
        states = ef.leaky_state(vectors, alpha)
        s = np.zeros(vectors.shape[1])
        for k in range(len(vectors)):
            s = alpha * s + vectors[k]
            np.testing.assert_allclose(states[k], s, atol=1e-12)

    def test_masses_scale_state_linearly(self, vectors):
        m = np.arange(1, len(vectors) + 1, dtype=float)
        s1 = ef.leaky_state(vectors, 0.85, masses=m)
        s2 = ef.leaky_state(vectors, 0.85, masses=2 * m)
        np.testing.assert_allclose(s2, 2 * s1, atol=1e-12)

    def test_unnormalized(self, vectors):
        states = ef.leaky_state(vectors, 0.85)
        norms = np.linalg.norm(states, axis=1)
        assert not np.allclose(norms, 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ef.leaky_state(np.zeros((0, 4)), 0.85)

    def test_bad_mass_shape_raises(self, vectors):
        with pytest.raises(ValueError):
            ef.leaky_state(vectors, 0.85, masses=np.ones(3))

    def test_alpha_out_of_domain_raises(self, vectors):
        """Documented domain is (0, 1]; enforce it for state and readout."""
        for bad in [0.0, -0.5, 1.5]:
            with pytest.raises(ValueError):
                ef.leaky_state(vectors, alpha=bad)
            with pytest.raises(ValueError):
                ef.trajectory(vectors, alpha=bad)
        ef.trajectory(vectors, alpha=1.0)  # boundary is valid (running mean)


class TestTrajectory:
    def test_is_normalized_leaky_state(self, vectors):
        states = ef.leaky_state(vectors, 0.85)
        traj = ef.trajectory(vectors, 0.85)
        expected = states / np.linalg.norm(states, axis=1, keepdims=True)
        np.testing.assert_allclose(traj, expected, atol=1e-12)

    def test_matches_smooth_exponential(self, vectors):
        """Readout direction == smooth_exponential output (denom is scalar)."""
        np.testing.assert_allclose(
            ef.trajectory(vectors, 0.85),
            ef.smooth_exponential(vectors, 0.85),
            atol=1e-9,
        )

    def test_mass_scaling_invariance(self, vectors):
        """Doubling all masses leaves the normalized readout unchanged."""
        m = np.arange(1, len(vectors) + 1, dtype=float)
        a = ef.trajectory(vectors, 0.85, masses=m)
        b = ef.trajectory(vectors, 0.85, masses=2 * m)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_alpha_near_one_approaches_running_mean(self, vectors):
        """Alpha convention: trajectory(E, alpha=0.999) approx running mean."""
        traj = ef.trajectory(vectors, 0.999)
        uni = ef.smooth_uniform(vectors)
        cos = (traj * uni).sum(axis=1)
        assert np.all(cos > 0.9999)

    def test_mass_changes_direction(self, vectors):
        m = np.ones(len(vectors))
        m[3] = 50.0
        a = ef.trajectory(vectors, 0.85)
        b = ef.trajectory(vectors, 0.85, masses=m)
        assert not np.allclose(a, b, atol=1e-3)


class TestPermutationProperties:
    def test_uniform_readout_permutation_invariant(self, vectors):
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(vectors))
        a = ef.weighted_mean(vectors)
        b = ef.weighted_mean(vectors[perm])
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_exponential_readout_not_invariant_and_alpha_monotone(self, vectors):
        """Exponential readout is order-sensitive, more so as alpha decreases."""
        rng = np.random.default_rng(1)

        def mean_perturbation(alpha, n_perms=20):
            base = ef.trajectory(vectors, alpha)[-1]
            d = []
            for _ in range(n_perms):
                p = rng.permutation(len(vectors))
                d.append(1 - float(base @ ef.trajectory(vectors[p], alpha)[-1]))
            return np.mean(d)

        d_low = mean_perturbation(0.5)
        d_high = mean_perturbation(0.9)
        assert d_low > 1e-4          # not invariant
        assert d_low > d_high        # more sensitive at lower alpha


class TestTimeDecayMassComposition:
    def test_masses_multiply_into_time_decay_weights(self, vectors):
        """Unified mass + time decay: weights compose by numpy *."""
        times = np.linspace(0, 5000, len(vectors))
        m = np.arange(1, len(vectors) + 1, dtype=float)
        traj = ef.smooth_time_decay(vectors, times, 3600, masses=m)
        for k in range(len(vectors)):
            w = ef.time_decay_weights(times[: k + 1], 3600) * m[: k + 1]
            expected = ef.weighted_mean(vectors[: k + 1], w)
            np.testing.assert_allclose(traj[k], expected, atol=1e-9)

    def test_no_masses_unchanged(self, vectors):
        times = np.linspace(0, 5000, len(vectors))
        a = ef.smooth_time_decay(vectors, times, 3600)
        b = ef.smooth_time_decay(vectors, times, 3600, masses=np.ones(len(vectors)))
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestHalfLife:
    def test_round_trip(self):
        for alpha in [0.5, 0.85, 0.99]:
            h = ef.alpha_to_half_life(alpha)
            np.testing.assert_allclose(ef.half_life_to_alpha(h), alpha, atol=1e-12)

    def test_alpha_half_gives_one_step(self):
        assert abs(ef.alpha_to_half_life(0.5) - 1.0) < 1e-12

    def test_alpha_one_is_infinite_memory(self):
        assert ef.alpha_to_half_life(1.0) == np.inf
        assert ef.half_life_to_alpha(np.inf) == 1.0

    def test_out_of_range_raises(self):
        for bad in [0.0, -0.1, 1.5]:
            with pytest.raises(ValueError):
                ef.alpha_to_half_life(bad)
        with pytest.raises(ValueError):
            ef.half_life_to_alpha(0.0)
