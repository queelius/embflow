"""Tests for motion operators ported from the embedding-dynamics motionlib."""
import json
from pathlib import Path

import numpy as np
import pytest

import embflow as ef

FIXTURE = Path(__file__).parent / "fixtures" / "motionlib_fixture.json"


@pytest.fixture
def vectors():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((20, 16))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class TestTurningCosines:
    def test_shape(self, vectors):
        t = ef.turning_cosines(vectors)
        assert t.shape == (len(vectors) - 2,)

    def test_straight_line_is_plus_one(self):
        line = np.array([[float(k), 0.0] for k in range(6)])
        np.testing.assert_allclose(ef.turning_cosines(line), 1.0, atol=1e-12)

    def test_reversal_is_minus_one(self):
        zigzag = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(ef.turning_cosines(zigzag), -1.0, atol=1e-12)

    def test_duplicate_points_give_nan(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        t = ef.turning_cosines(path)
        assert np.isnan(t[0]) and np.isnan(t[1])

    def test_translation_invariant(self, vectors):
        c = np.full(vectors.shape[1], 3.7)
        a = ef.turning_cosines(vectors)
        b = ef.turning_cosines(vectors + c)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_short_input_empty(self):
        assert ef.turning_cosines(np.zeros((2, 4))).size == 0
        assert ef.turning_cosines(np.zeros((1, 4))).size == 0

    def test_distinct_from_angular_velocity(self, vectors):
        """turning_cosines: cos between VELOCITIES; angular_velocity: 1-cos between POINTS."""
        t = ef.turning_cosines(vectors)
        a = ef.angular_velocity(vectors)
        assert t.shape[0] == a.shape[0] - 1
        assert not np.allclose(1 - t, a[1:], atol=0.1)


class TestTortuosity:
    def test_straight_line_is_one(self):
        line = np.array([[float(k), 0.0] for k in range(12)])
        assert abs(ef.tortuosity(line) - 1.0) < 1e-9

    def test_backtracking_below_straight(self, vectors):
        line = np.array([[float(k), 0.0] for k in range(12)])
        zig = np.array([[float(k % 2), 0.0] for k in range(12)])
        assert ef.tortuosity(zig, window=4) < ef.tortuosity(line, window=4)

    def test_too_short_is_nan(self):
        assert np.isnan(ef.tortuosity(np.zeros((2, 3))))

    def test_window_clamped_to_length(self):
        line = np.array([[float(k), 0.0] for k in range(5)])
        assert abs(ef.tortuosity(line, window=100) - 1.0) < 1e-9

    def test_stationary_is_nan(self):
        const = np.tile([1.0, 0.0], (10, 1))
        assert np.isnan(ef.tortuosity(const))


class TestSpeedAutocorr:
    def test_alternating_steps_negative_lag1(self):
        # Step sizes alternate 1.0, 0.1: lag-1 autocorrelation ~ -1.
        xs = np.cumsum([0.0] + [1.0, 0.1] * 6)
        traj = np.stack([xs, np.zeros_like(xs)], axis=1)
        assert ef.speed_autocorr(traj, lag=1) < -0.9

    def test_alternating_steps_positive_lag2(self):
        xs = np.cumsum([0.0] + [1.0, 0.1] * 6)
        traj = np.stack([xs, np.zeros_like(xs)], axis=1)
        assert ef.speed_autocorr(traj, lag=2) > 0.9

    def test_constant_speed_is_nan(self):
        line = np.array([[float(k), 0.0] for k in range(10)])
        assert np.isnan(ef.speed_autocorr(line))

    def test_too_short_is_nan(self):
        assert np.isnan(ef.speed_autocorr(np.zeros((3, 2))))


class TestAdaptiveAlphaConvention:
    def test_slow_drift_prefers_long_memory(self):
        """A sequence hugging a fixed direction with noise: the running
        mean is the best predictor, so alpha should land high."""
        rng = np.random.default_rng(5)
        base = np.array([1.0] + [0.0] * 15)
        vecs = base + 0.2 * rng.standard_normal((60, 16))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        assert ef.adaptive_alpha(vecs) >= 0.9

    def test_jumpy_sequence_prefers_short_memory(self):
        """Blockwise topic jumps: old state mispredicts, alpha lands low."""
        rng = np.random.default_rng(6)
        blocks = []
        for b in range(10):
            e = np.zeros(16)
            e[b] = 1.0
            blocks.append(e + 0.05 * rng.standard_normal((3, 16)))
        vecs = np.concatenate(blocks)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        assert ef.adaptive_alpha(vecs) <= 0.6


def _fixture_input(fix):
    rng = np.random.default_rng(fix["seed"])
    E = rng.standard_normal((fix["n"], fix["d"]))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


class TestMotionSignature:
    def test_keys_and_types(self, vectors):
        sig = ef.motion_signature(vectors)
        assert set(sig) == {
            "speed_mean", "speed_std", "turn_mean", "turn_std",
            "speed_ac1", "tortuosity_w8", "alpha_hat",
        }
        assert all(isinstance(v, float) for v in sig.values())

    def test_window_in_key(self, vectors):
        sig = ef.motion_signature(vectors, window=4)
        assert "tortuosity_w4" in sig

    def test_without_alpha(self, vectors):
        assert "alpha_hat" not in ef.motion_signature(vectors, with_alpha=False)


class TestMotionlibFidelity:
    """Acceptance criterion 4: outputs match motionlib on the same input."""

    @pytest.fixture(scope="class")
    def fix(self):
        return json.loads(FIXTURE.read_text())

    @pytest.fixture(scope="class")
    def E(self, fix):
        return _fixture_input(fix)

    def test_motion_signature_matches_motion_scalars(self, fix, E):
        sig = ef.motion_signature(E, window=8, with_alpha=True)
        for key, expected in fix["motion_scalars"].items():
            np.testing.assert_allclose(sig[key], expected, atol=1e-9, err_msg=key)

    def test_turning_cosines(self, fix, E):
        np.testing.assert_allclose(
            ef.turning_cosines(E)[:5], fix["turning_cosines_first5"], atol=1e-9
        )

    def test_tortuosity(self, fix, E):
        np.testing.assert_allclose(ef.tortuosity(E, 8), fix["tortuosity_w8"], atol=1e-9)

    def test_speeds(self, fix, E):
        np.testing.assert_allclose(ef.speed(E)[:5], fix["speeds_first5"], atol=1e-9)

    def test_speed_autocorr(self, fix, E):
        np.testing.assert_allclose(
            ef.speed_autocorr(E, lag=1), fix["lag1_autocorr_speeds"], atol=1e-9
        )

    def test_adaptive_alpha(self, fix, E):
        np.testing.assert_allclose(ef.adaptive_alpha(E), fix["adaptive_alpha"], atol=1e-12)

    def test_trajectory_matches_smoothed_trajectory(self, fix, E):
        np.testing.assert_allclose(
            ef.trajectory(E, 0.85)[-1],
            fix["smoothed_trajectory_alpha085_last"],
            atol=1e-6,  # motionlib normalizes with +EPS, embflow with where>0
        )
