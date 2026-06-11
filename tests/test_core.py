"""Core tests for embflow: weights, smoothers, differential operators, comparison."""
import numpy as np
import pytest

import embflow as ef


@pytest.fixture
def vectors():
    """10 random unit vectors in 8 dimensions."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((10, 8))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def meta():
    """Metadata for 10 items with role and timestamp."""
    return [
        {"role": "user", "timestamp": 1000.0},
        {"role": "assistant", "timestamp": 1001.0},
        {"role": "user", "timestamp": 1005.0},
        {"role": "assistant", "timestamp": 1006.0},
        {"role": "user", "timestamp": 1020.0},
        {"role": "assistant", "timestamp": 1021.0},
        {"role": "user", "timestamp": 1100.0},
        {"role": "assistant", "timestamp": 1101.0},
        {"role": "user", "timestamp": 1200.0},
        {"role": "assistant", "timestamp": 1201.0},
    ]


# === Fold-style: weighted_mean + weight generators ===

class TestWeightedMean:
    def test_uniform_is_normalized_mean(self, vectors):
        result = ef.weighted_mean(vectors, ef.uniform_weights(len(vectors)))
        expected = vectors.mean(axis=0)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_no_weights_defaults_to_uniform(self, vectors):
        np.testing.assert_allclose(
            ef.weighted_mean(vectors),
            ef.weighted_mean(vectors, ef.uniform_weights(len(vectors))),
            atol=1e-6,
        )

    def test_returns_unit_vector(self, vectors):
        for w in [
            ef.uniform_weights(len(vectors)),
            ef.exponential_weights(len(vectors), 0.85),
            ef.novelty_weights(vectors),
        ]:
            result = ef.weighted_mean(vectors, w)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_exponential_alpha_1_equals_uniform(self, vectors):
        exp = ef.weighted_mean(vectors, ef.exponential_weights(len(vectors), 1.0))
        uni = ef.weighted_mean(vectors, ef.uniform_weights(len(vectors)))
        np.testing.assert_allclose(exp, uni, atol=1e-6)

    def test_low_alpha_favors_last(self, vectors):
        result = ef.weighted_mean(
            vectors, ef.exponential_weights(len(vectors), 0.01)
        )
        last = vectors[-1] / np.linalg.norm(vectors[-1])
        assert np.dot(result, last) > 0.99

    def test_reverse_low_alpha_favors_first(self, vectors):
        result = ef.weighted_mean(
            vectors, ef.reverse_exponential_weights(len(vectors), 0.01)
        )
        first = vectors[0] / np.linalg.norm(vectors[0])
        assert np.dot(result, first) > 0.99

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ef.weighted_mean(np.zeros((0, 4)))


class TestWeightComposition:
    """Weight arrays form a commutative monoid under pointwise multiplication."""

    def test_uniform_is_identity(self, vectors):
        exp_w = ef.exponential_weights(len(vectors), 0.85)
        ones = ef.uniform_weights(len(vectors))
        a = ef.weighted_mean(vectors, exp_w)
        b = ef.weighted_mean(vectors, ones * exp_w)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_multiplication_is_commutative(self, vectors, meta):
        a = ef.exponential_weights(len(vectors), 0.85)
        b = ef.field_weights(meta, "role", {"user": 3.0})
        ab = ef.weighted_mean(vectors, a * b)
        ba = ef.weighted_mean(vectors, b * a)
        np.testing.assert_allclose(ab, ba, atol=1e-6)

    def test_multiplication_is_associative(self, vectors, meta):
        a = ef.exponential_weights(len(vectors), 0.85)
        b = ef.field_weights(meta, "role", {"user": 3.0})
        c = ef.gaussian_weights(len(vectors), focus=0.8)
        left = ef.weighted_mean(vectors, (a * b) * c)
        right = ef.weighted_mean(vectors, a * (b * c))
        np.testing.assert_allclose(left, right, atol=1e-6)

    def test_composed_differs_from_parts(self, vectors, meta):
        exp_w = ef.exponential_weights(len(vectors), 0.85)
        role_w = ef.field_weights(meta, "role", {"user": 3.0, "assistant": 0.5})
        exp_proj = ef.weighted_mean(vectors, exp_w)
        role_proj = ef.weighted_mean(vectors, role_w)
        combined = ef.weighted_mean(vectors, exp_w * role_w)
        assert not np.allclose(combined, exp_proj, atol=1e-3)
        assert not np.allclose(combined, role_proj, atol=1e-3)


class TestFieldWeights:
    def test_changes_result(self, vectors, meta):
        uniform = ef.weighted_mean(vectors, ef.uniform_weights(len(vectors)))
        weighted = ef.weighted_mean(
            vectors,
            ef.field_weights(meta, "role", {"user": 5.0, "assistant": 0.1}),
        )
        assert not np.allclose(uniform, weighted, atol=1e-3)

    def test_no_mapping_uses_numeric_value(self):
        meta = [{"x": 2.0}, {"x": 3.0}, {"x": 0.5}]
        w = ef.field_weights(meta, "x")
        np.testing.assert_allclose(w, [2.0, 3.0, 0.5])

    def test_missing_field_defaults_to_one(self):
        meta = [{"x": 2.0}, {}, {"x": 3.0}]
        w = ef.field_weights(meta, "x")
        np.testing.assert_allclose(w, [2.0, 1.0, 3.0])

    def test_mapping_unmapped_uses_default(self):
        m = [{"role": "user"}, {"role": "assistant"}, {"role": "system"}]
        w = ef.field_weights(m, "role", {"user": 3.0, "assistant": 1.0}, default=0.1)
        np.testing.assert_allclose(w, [3.0, 1.0, 0.1])


class TestTimeDecayWeights:
    def test_decay_matches_half_life(self):
        w = ef.time_decay_weights([0.0, 3600.0], half_life_seconds=3600)
        assert abs(w[1] - 1.0) < 1e-6
        assert abs(w[0] - 0.5) < 1e-6

    def test_future_timestamp_does_not_silently_peak(self):
        # Non-monotone times: the latest observed is at index 1, not the
        # final entry. t_ref must be max(times), not times[-1].
        w = ef.time_decay_weights([1000.0, 3000.0, 500.0], half_life_seconds=3600)
        assert np.argmax(w) == 1
        assert w[2] < 1.0 - 1e-6

    def test_empty_returns_empty(self):
        w = ef.time_decay_weights([], half_life_seconds=3600)
        assert len(w) == 0


class TestNoveltyWeights:
    def test_first_weight_is_one(self, vectors):
        w = ef.novelty_weights(vectors)
        assert w[0] == 1.0

    def test_constant_sequence_floors_at_epsilon(self):
        const = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        w = ef.novelty_weights(const)
        assert w[0] == 1.0
        np.testing.assert_allclose(w[1:], 0.01)

    def test_anti_aligned_gets_large_weight(self):
        vecs = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        w = ef.novelty_weights(vecs)
        assert w[2] > 1.5  # approaches 2.0


class TestGaussianWeights:
    def test_peak_at_midpoint(self):
        w = ef.gaussian_weights(11, focus=0.5)
        assert np.argmax(w) == 5

    def test_fractional_endpoints(self):
        assert np.argmax(ef.gaussian_weights(10, focus=0.0)) == 0
        assert np.argmax(ef.gaussian_weights(10, focus=1.0)) == 9

    def test_absolute_index_via_int(self):
        # Int focus is treated as absolute index.
        w = ef.gaussian_weights(10, focus=3, sigma=1.0)
        assert np.argmax(w) == 3


# === Scan-style: smoothers ===

class TestSmoothExponential:
    def test_shape_matches_input(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        assert traj.shape == vectors.shape

    def test_first_equals_unit_first(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        np.testing.assert_allclose(
            traj[0], vectors[0] / np.linalg.norm(vectors[0]), atol=1e-6
        )

    def test_last_equals_weighted_mean(self, vectors):
        alpha = 0.85
        traj = ef.smooth_exponential(vectors, alpha)
        expected = ef.weighted_mean(
            vectors, ef.exponential_weights(len(vectors), alpha)
        )
        np.testing.assert_allclose(traj[-1], expected, atol=1e-6)

    def test_matches_prefix_weighted_mean(self, vectors):
        """O(n) online form matches the explicit O(n^2) prefix formula."""
        alpha = 0.85
        traj = ef.smooth_exponential(vectors, alpha)
        for j in range(len(vectors)):
            w = ef.exponential_weights(j + 1, alpha)
            expected = ef.weighted_mean(vectors[:j + 1], w)
            np.testing.assert_allclose(traj[j], expected, atol=1e-6)

    def test_alpha_1_is_running_mean(self, vectors):
        # alpha=1 degenerates to the uniform running mean.
        exp_traj = ef.smooth_exponential(vectors, 1.0)
        uni_traj = ef.smooth_uniform(vectors)
        np.testing.assert_allclose(exp_traj, uni_traj, atol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ef.smooth_exponential(np.zeros((0, 4)), 0.85)


class TestSmoothUniform:
    def test_running_mean(self, vectors):
        traj = ef.smooth_uniform(vectors)
        for j in range(len(vectors)):
            expected = vectors[:j + 1].mean(axis=0)
            expected = expected / np.linalg.norm(expected)
            np.testing.assert_allclose(traj[j], expected, atol=1e-6)


# === Differential operators ===

class TestVelocity:
    def test_shape(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        vel = ef.velocity(traj)
        assert vel.shape == (len(vectors) - 1, vectors.shape[1])

    def test_constant_has_zero_velocity(self):
        const = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        np.testing.assert_allclose(ef.velocity(const), 0, atol=1e-10)

    def test_translation_invariant(self, vectors):
        """velocities(E + c) == velocities(E)."""
        c = np.full(vectors.shape[1], 2.5)
        np.testing.assert_allclose(
            ef.velocity(vectors + c), ef.velocity(vectors), atol=1e-9
        )


class TestCurvature:
    def test_shape(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        curv = ef.curvature(traj)
        assert curv.shape == (len(vectors) - 2, vectors.shape[1])

    def test_linear_has_zero_curvature(self):
        line = np.array([[k, 0.0, 0.0, 0.0] for k in range(5)], dtype=float)
        np.testing.assert_allclose(ef.curvature(line), 0, atol=1e-10)


class TestJerk:
    def test_shape(self, vectors):
        j = ef.jerk(vectors)
        assert j.shape == (len(vectors) - 3, vectors.shape[1])

    def test_constant_has_zero_jerk(self):
        const = np.tile([1.0, 0.0, 0.0, 0.0], (6, 1))
        np.testing.assert_allclose(ef.jerk(const), 0, atol=1e-10)

    def test_quadratic_has_zero_jerk(self):
        # f(k) = k^2 * [1, 0, 0]: constant second diff, zero third diff.
        traj = np.array([[k ** 2, 0.0, 0.0] for k in range(6)], dtype=float)
        np.testing.assert_allclose(ef.jerk(traj), 0, atol=1e-10)


class TestSpeed:
    def test_shape_and_nonneg(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        s = ef.speed(traj)
        assert s.shape == (len(vectors) - 1,)
        assert np.all(s >= 0)


class TestAngularVelocity:
    def test_shape(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        a = ef.angular_velocity(traj)
        assert a.shape == (len(vectors) - 1,)

    def test_zero_for_constant_direction(self):
        # Unit vectors in the same direction: zero angular velocity.
        same = np.tile([1.0, 0.0], (5, 1))
        np.testing.assert_allclose(ef.angular_velocity(same), 0, atol=1e-10)


# === Global and second-order geometry ===

class TestDrift:
    def test_zero_for_constant(self):
        const = np.tile([1.0, 0.0, 0.0], (5, 1))
        assert ef.drift(const) < 1e-6

    def test_positive_for_changing(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        assert ef.drift(traj) > 0

    def test_max_for_opposite(self):
        opposite = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert abs(ef.drift(opposite) - 2.0) < 1e-6


class TestArcLength:
    def test_starts_at_zero(self, vectors):
        assert ef.arc_length(vectors)[0] == 0.0

    def test_monotonically_nondecreasing(self, vectors):
        al = ef.arc_length(vectors)
        assert np.all(np.diff(al) >= 0)

    def test_length_one_returns_zeros(self):
        assert ef.arc_length(np.array([[1.0, 0.0]])).tolist() == [0.0]

    def test_straight_line_sums_segments(self):
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        np.testing.assert_allclose(ef.arc_length(traj), [0.0, 1.0, 3.0])


class TestLocalCurvatureRadius:
    def test_shape(self, vectors):
        r = ef.local_curvature_radius(vectors)
        assert r.shape == (len(vectors) - 2,)

    def test_collinear_is_infinite(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert np.all(np.isinf(ef.local_curvature_radius(line)))

    def test_unit_circle_quarter(self):
        # Three points on a unit circle: circumradius = 1.0.
        t = np.array([
            [1.0, 0.0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0.0, 1.0],
        ])
        r = ef.local_curvature_radius(t)
        assert abs(r[0] - 1.0) < 1e-6

    def test_short_trajectory_returns_empty(self):
        assert ef.local_curvature_radius(np.array([[1.0, 0.0]])).size == 0
        assert ef.local_curvature_radius(
            np.array([[1.0, 0.0], [0.0, 1.0]])
        ).size == 0


class TestVelocityCovariance:
    def test_shape(self, vectors):
        cov = ef.velocity_covariance(vectors, window=3)
        d = vectors.shape[1]
        assert cov.shape == (len(vectors) - 1, d, d)

    def test_symmetric(self, vectors):
        cov = ef.velocity_covariance(vectors, window=2)
        for c in cov:
            np.testing.assert_allclose(c, c.T, atol=1e-10)

    def test_unidirectional_motion_is_rank_one(self):
        # Velocity is the same vector every step: covariance is rank-1.
        direction = np.array([1.0, 0.0, 0.0])
        traj = np.array([k * direction for k in range(6)], dtype=float)
        cov = ef.velocity_covariance(traj, window=5)
        eigvals = np.linalg.eigvalsh(cov[2])
        assert eigvals[-1] > 0.5
        assert np.all(np.abs(eigvals[:-1]) < 1e-9)


# === Segmentation ===

class TestPeaks:
    def test_basic(self):
        signal = np.array([0.1, 0.5, 0.1, 0.1, 0.3, 0.1])
        assert set(ef.peaks(signal, threshold=0.2)) == {1, 4}

    def test_prefers_strongest_with_min_distance(self):
        # Two candidates above threshold: index 1 (0.3) and index 4 (0.5).
        # With min_distance=4, the STRONGER one must win.
        signal = np.array([0.1, 0.3, 0.1, 0.1, 0.5, 0.1])
        result = ef.peaks(signal, threshold=0.2, min_distance=4)
        assert 4 in result
        assert 1 not in result

    def test_empty(self):
        assert ef.peaks(np.array([])) == []

    def test_below_threshold(self):
        assert ef.peaks(np.zeros(10), threshold=0.5) == []

    def test_returns_sorted(self):
        signal = np.array([0.5, 0.1, 0.4, 0.1, 0.6, 0.1, 0.3])
        result = ef.peaks(signal, threshold=0.2, min_distance=2)
        assert result == sorted(result)


class TestSegmentation:
    def test_segment_preserves_data(self, vectors):
        segs = ef.segment(vectors, [3, 7])
        assert sum(len(s["vectors"]) for s in segs) == len(vectors)

    def test_segment_no_boundaries(self, vectors):
        segs = ef.segment(vectors, [])
        assert len(segs) == 1
        assert len(segs[0]["vectors"]) == len(vectors)

    def test_auto_segment_returns_segments(self, vectors):
        segs = ef.auto_segment(vectors, alpha=0.85)
        assert len(segs) >= 1
        assert sum(len(s["vectors"]) for s in segs) == len(vectors)


class TestAutoSegmentWindow:
    def test_window_method_detects_clear_split(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((30, 16)) * 0.05 + np.array([1.0] + [0.0] * 15)
        b = rng.standard_normal((30, 16)) * 0.05 + np.array([0.0, 1.0] + [0.0] * 14)
        vecs = np.concatenate([a, b], axis=0)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        segs = ef.auto_segment(vecs, threshold="window", window_size=10)
        assert len(segs) >= 2
        assert sum(len(s["vectors"]) for s in segs) == len(vecs)

    def test_window_method_flat_input_no_splits(self):
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((40, 8)) * 0.01 + np.array([1.0] + [0.0] * 7)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        segs = ef.auto_segment(vecs, threshold="window", window_size=8)
        assert len(segs) == 1


# === Adaptive analysis ===

class TestAdaptiveAlpha:
    def test_returns_grid_value(self, vectors):
        alpha = ef.adaptive_alpha(vectors)
        assert isinstance(alpha, float)
        assert any(abs(alpha - g) < 1e-12 for g in ef.ALPHA_GRID)

    def test_short_sequence_is_nan(self):
        assert np.isnan(ef.adaptive_alpha(np.zeros((2, 4))))

    def test_constant_sequence_prefers_longest_memory(self):
        # Every alpha predicts a constant sequence almost perfectly; the
        # EPS-guarded normalization scores norm/(norm+EPS), and higher
        # alpha accumulates a larger-norm state, so near-ties resolve
        # toward the LONGEST memory (verified identical to motionlib).
        const = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        assert abs(ef.adaptive_alpha(const) - ef.ALPHA_GRID[-1]) < 1e-12

    def test_custom_grid(self, vectors):
        grid = np.array([0.5, 0.9])
        assert ef.adaptive_alpha(vectors, grid=grid) in (0.5, 0.9)

    def test_max_messages_cap(self):
        rng = np.random.default_rng(3)
        vecs = rng.standard_normal((50, 8))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        capped = ef.adaptive_alpha(vecs, max_messages=20)
        truncated = ef.adaptive_alpha(vecs[:20])
        assert capped == truncated


class TestStructuralRichness:
    def test_returns_float(self, vectors):
        r = ef.structural_richness(vectors)
        assert isinstance(r, float)
        assert r >= 0

    def test_constant_has_zero_richness(self):
        const = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        assert ef.structural_richness(const) < 1e-6

    def test_custom_weight_fns(self, vectors):
        # Passing a single weight function yields zero richness (pairs require 2+).
        r = ef.structural_richness(
            vectors, weight_fns=[lambda v, m: ef.uniform_weights(len(v))]
        )
        assert r == 0.0


# === Comparison ===

class TestTrajectoryDistance:
    def test_self_distance_is_zero(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        for method in ["dtw", "resample", "frechet"]:
            d = ef.trajectory_distance(traj, traj, method=method)
            assert d < 1e-6, f"self-distance nonzero for {method}"

    def test_different_trajectories_positive(self, vectors):
        rng = np.random.default_rng(99)
        other = rng.standard_normal(vectors.shape)
        other /= np.linalg.norm(other, axis=1, keepdims=True)

        a = ef.smooth_exponential(vectors, 0.85)
        b = ef.smooth_exponential(other, 0.85)
        for method in ["dtw", "resample", "frechet"]:
            d = ef.trajectory_distance(a, b, method=method)
            assert d > 0, f"different trajectories have zero distance for {method}"

    def test_different_lengths(self):
        a = np.random.default_rng(1).standard_normal((5, 4))
        b = np.random.default_rng(2).standard_normal((15, 4))
        for method in ["dtw", "resample", "frechet"]:
            assert ef.trajectory_distance(a, b, method=method) >= 0


class TestShapeAndEndpointDistance:
    def test_shape_self_distance_small(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        assert ef.trajectory_distance(traj, traj, method="shape") < 1e-6

    def test_endpoint_self_distance_zero(self, vectors):
        traj = ef.smooth_exponential(vectors, 0.85)
        assert ef.trajectory_distance(traj, traj, method="endpoint") < 1e-6

    def test_shape_short_trajectory_returns_one(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert ef.trajectory_distance(a, a, method="shape") == 1.0


class TestVelocityGram:
    def test_shape(self, vectors):
        G = ef.velocity_gram(vectors)
        assert G.shape == (len(vectors) - 1, len(vectors) - 1)

    def test_diag_is_squared_speed(self, vectors):
        G = ef.velocity_gram(vectors)
        np.testing.assert_allclose(np.diag(G), ef.speed(vectors) ** 2, atol=1e-9)

    def test_rotation_and_translation_invariant(self, vectors):
        rng = np.random.default_rng(5)
        Q, _ = np.linalg.qr(rng.standard_normal((vectors.shape[1],) * 2))
        moved = vectors @ Q.T + 0.7
        np.testing.assert_allclose(
            ef.velocity_gram(moved), ef.velocity_gram(vectors), atol=1e-9
        )

    def test_permutation_sensitive(self, vectors):
        perm = np.random.default_rng(6).permutation(len(vectors))
        assert not np.allclose(
            ef.velocity_gram(vectors[perm]), ef.velocity_gram(vectors), atol=1e-3
        )


class TestContinuationScore:
    def test_identical_sequence_high_score(self, vectors):
        assert ef.continuation_score(vectors, vectors) > 0.5

    def test_returns_float(self, vectors):
        other = np.random.default_rng(99).standard_normal(vectors.shape)
        score = ef.continuation_score(vectors, other)
        assert isinstance(score, float)
        assert -1 <= score <= 1


class TestPublicApi:
    def test_version(self):
        assert ef.__version__ == "0.3.0"

    def test_new_symbols_exported(self):
        for name in [
            "leaky_state", "trajectory", "alpha_to_half_life",
            "half_life_to_alpha", "turning_cosines", "tortuosity",
            "speed_autocorr", "motion_signature", "ALPHA_GRID",
            "shuffle", "role_slot_shuffle", "null_corrected",
            "paired_stats", "velocity_gram", "lens_weights",
            "default_lenses", "prefix_experiment", "EmbedFn",
            "cached_embed_fn", "openai_embed_fn", "ollama_embed_fn",
        ]:
            assert hasattr(ef, name), name


# === Edge-case guards ===

class TestEmptyInput:
    def test_weighted_mean_empty_raises(self):
        with pytest.raises(ValueError):
            ef.weighted_mean(np.zeros((0, 4)))

    def test_smoothers_empty_raises(self):
        empty = np.zeros((0, 4))
        with pytest.raises(ValueError):
            ef.smooth_exponential(empty, 0.85)
        with pytest.raises(ValueError):
            ef.smooth_uniform(empty)
        with pytest.raises(ValueError):
            ef.smooth_reverse_exponential(empty, 0.85)

    def test_trajectory_distance_empty_raises(self, vectors):
        empty = np.zeros((0, vectors.shape[1]))
        for method in ["dtw", "resample", "frechet", "shape", "endpoint"]:
            with pytest.raises(ValueError):
                ef.trajectory_distance(empty, vectors, method=method)
            with pytest.raises(ValueError):
                ef.trajectory_distance(vectors, empty, method=method)

    def test_continuation_score_empty_raises(self, vectors):
        empty = np.zeros((0, vectors.shape[1]))
        with pytest.raises(ValueError):
            ef.continuation_score(empty, vectors)
        with pytest.raises(ValueError):
            ef.continuation_score(vectors, empty)

    def test_structural_richness_empty_raises(self):
        with pytest.raises(ValueError):
            ef.structural_richness(np.zeros((0, 4)))
