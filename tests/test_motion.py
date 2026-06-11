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
