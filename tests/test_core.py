"""Core tests for embflow: lenses, operations, and comparison."""
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


# === Lens basics ===

class TestLensProject:
    def test_uniform_is_mean(self, vectors):
        result = ef.Uniform().project(vectors)
        expected = vectors.mean(axis=0)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_project_returns_unit_vector(self, vectors):
        for lens in [ef.Uniform(), ef.Exponential(0.85), ef.Surprise()]:
            result = lens.project(vectors)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_exponential_alpha_1_is_uniform(self, vectors):
        exp = ef.Exponential(alpha=1.0).project(vectors)
        uni = ef.Uniform().project(vectors)
        np.testing.assert_allclose(exp, uni, atol=1e-6)

    def test_exponential_low_alpha_favors_last(self, vectors):
        result = ef.Exponential(alpha=0.01).project(vectors)
        last = vectors[-1] / np.linalg.norm(vectors[-1])
        # Should be very close to last vector
        sim = np.dot(result, last)
        assert sim > 0.99

    def test_reverse_exponential_low_alpha_favors_first(self, vectors):
        result = ef.ReverseExponential(alpha=0.01).project(vectors)
        first = vectors[0] / np.linalg.norm(vectors[0])
        sim = np.dot(result, first)
        assert sim > 0.99


class TestLensTrajectory:
    def test_trajectory_length_matches_input(self, vectors):
        traj = ef.Exponential(0.85).trajectory(vectors)
        assert traj.shape == vectors.shape

    def test_trajectory_last_equals_project(self, vectors):
        lens = ef.Exponential(0.85)
        traj = lens.trajectory(vectors)
        proj = lens.project(vectors)
        np.testing.assert_allclose(traj[-1], proj, atol=1e-6)

    def test_trajectory_first_is_first_vector(self, vectors):
        traj = ef.Uniform().trajectory(vectors)
        first = vectors[0] / np.linalg.norm(vectors[0])
        np.testing.assert_allclose(traj[0], first, atol=1e-6)

    def test_uniform_trajectory_is_running_mean(self, vectors):
        traj = ef.Uniform().trajectory(vectors)
        for j in range(len(vectors)):
            expected = vectors[:j + 1].mean(axis=0)
            expected = expected / np.linalg.norm(expected)
            np.testing.assert_allclose(traj[j], expected, atol=1e-6)


class TestLensComposition:
    def test_uniform_is_identity(self, vectors):
        lens = ef.Exponential(0.85)
        composed = ef.Uniform() * lens
        a = lens.project(vectors)
        b = composed.project(vectors)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_composition_is_commutative(self, vectors, meta):
        a = ef.Exponential(0.85)
        b = ef.FieldWeight("role", {"user": 3.0})
        ab = (a * b).project(vectors, meta)
        ba = (b * a).project(vectors, meta)
        np.testing.assert_allclose(ab, ba, atol=1e-6)

    def test_composition_is_associative(self, vectors, meta):
        a = ef.Exponential(0.85)
        b = ef.FieldWeight("role", {"user": 3.0})
        c = ef.Gaussian(focus=0.8)
        ab_c = ((a * b) * c).project(vectors, meta)
        a_bc = (a * (b * c)).project(vectors, meta)
        np.testing.assert_allclose(ab_c, a_bc, atol=1e-6)

    def test_composed_differs_from_parts(self, vectors, meta):
        exp = ef.Exponential(0.85).project(vectors)
        role = ef.FieldWeight("role", {"user": 3.0, "assistant": 0.5}).project(vectors, meta)
        combined = (ef.Exponential(0.85) * ef.FieldWeight("role", {"user": 3.0, "assistant": 0.5})).project(vectors, meta)
        # Combined should differ from both parts
        assert not np.allclose(combined, exp, atol=1e-3)
        assert not np.allclose(combined, role, atol=1e-3)


class TestFieldWeight:
    def test_no_meta_returns_uniform(self, vectors):
        result = ef.FieldWeight("role", {"user": 3.0}).project(vectors)
        uniform = ef.Uniform().project(vectors)
        np.testing.assert_allclose(result, uniform, atol=1e-6)

    def test_field_weight_changes_result(self, vectors, meta):
        uniform = ef.Uniform().project(vectors)
        weighted = ef.FieldWeight("role", {"user": 5.0, "assistant": 0.1}).project(vectors, meta)
        assert not np.allclose(uniform, weighted, atol=1e-3)


class TestHalfLife:
    def test_exponential_half_life(self):
        lens = ef.Exponential(0.85)
        hl = lens.half_life()
        assert abs(hl - 4.265) < 0.01

    def test_uniform_no_half_life(self):
        assert ef.Uniform().half_life() is None


# === Operations ===

class TestVelocity:
    def test_velocity_shape(self, vectors):
        traj = ef.Exponential(0.85).trajectory(vectors)
        vel = ef.velocity(traj)
        assert vel.shape == (len(vectors) - 1, vectors.shape[1])

    def test_velocity_of_constant_is_zero(self):
        constant = np.tile([1, 0, 0, 0], (5, 1)).astype(float)
        vel = ef.velocity(constant)
        np.testing.assert_allclose(vel, 0, atol=1e-10)


class TestCurvature:
    def test_curvature_shape(self, vectors):
        traj = ef.Exponential(0.85).trajectory(vectors)
        curv = ef.curvature(traj)
        assert curv.shape == (len(vectors) - 2, vectors.shape[1])

    def test_curvature_of_linear_is_zero(self):
        # Linear trajectory: constant velocity
        linear = np.array([[i, 0, 0, 0] for i in range(5)], dtype=float)
        curv = ef.curvature(linear)
        np.testing.assert_allclose(curv, 0, atol=1e-10)


class TestDrift:
    def test_drift_zero_for_constant(self):
        constant = np.tile([1, 0, 0], (5, 1)).astype(float)
        assert ef.drift(constant) < 1e-6

    def test_drift_positive_for_changing(self, vectors):
        traj = ef.Exponential(0.85).trajectory(vectors)
        d = ef.drift(traj)
        assert d > 0

    def test_drift_max_for_opposite(self):
        opposite = np.array([[1, 0], [-1, 0]], dtype=float)
        d = ef.drift(opposite)
        assert abs(d - 2.0) < 1e-6  # cosine distance of opposite vectors


class TestSegmentation:
    def test_segment_preserves_data(self, vectors):
        segs = ef.segment(vectors, [3, 7])
        total = sum(len(s["vectors"]) for s in segs)
        assert total == len(vectors)

    def test_segment_no_boundaries(self, vectors):
        segs = ef.segment(vectors, [])
        assert len(segs) == 1
        assert len(segs[0]["vectors"]) == len(vectors)

    def test_auto_segment_returns_segments(self, vectors):
        segs = ef.auto_segment(vectors, alpha=0.85)
        assert len(segs) >= 1
        total = sum(len(s["vectors"]) for s in segs)
        assert total == len(vectors)


class TestAdaptiveAlpha:
    def test_returns_float(self, vectors):
        alpha = ef.adaptive_alpha(vectors)
        assert isinstance(alpha, float)
        assert 0 < alpha < 1


class TestStructuralRichness:
    def test_returns_float(self, vectors):
        r = ef.structural_richness(vectors)
        assert isinstance(r, float)
        assert r >= 0

    def test_constant_has_zero_richness(self):
        constant = np.tile([1, 0, 0, 0], (10, 1)).astype(float)
        r = ef.structural_richness(constant)
        assert r < 1e-6


# === Comparison ===

class TestTrajectoryDistance:
    def test_self_distance_is_zero(self, vectors):
        traj = ef.Exponential(0.85).trajectory(vectors)
        for method in ["dtw", "resample", "frechet"]:
            d = ef.trajectory_distance(traj, traj, method=method)
            assert d < 1e-6, f"Self-distance not zero for {method}"

    def test_different_trajectories_positive(self, vectors):
        rng = np.random.default_rng(99)
        other = rng.standard_normal(vectors.shape)
        norms = np.linalg.norm(other, axis=1, keepdims=True)
        other = other / norms

        traj_a = ef.Exponential(0.85).trajectory(vectors)
        traj_b = ef.Exponential(0.85).trajectory(other)

        for method in ["dtw", "resample", "frechet"]:
            d = ef.trajectory_distance(traj_a, traj_b, method=method)
            assert d > 0, f"Different trajectories have zero distance for {method}"

    def test_different_lengths(self):
        a = np.random.default_rng(1).standard_normal((5, 4))
        b = np.random.default_rng(2).standard_normal((15, 4))
        for method in ["dtw", "resample", "frechet"]:
            d = ef.trajectory_distance(a, b, method=method)
            assert d >= 0


class TestContinuationScore:
    def test_identical_sequence_high_score(self, vectors):
        score = ef.continuation_score(vectors, vectors)
        assert score > 0.5

    def test_returns_float(self, vectors):
        other = np.random.default_rng(99).standard_normal(vectors.shape)
        score = ef.continuation_score(vectors, other)
        assert isinstance(score, float)
        assert -1 <= score <= 1
