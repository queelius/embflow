"""Regenerate motionlib_fixture.json from the AUTHORITATIVE motionlib.

motionlib is the reference implementation in the embedding-dynamics
paper repo (battle-tested on 1,768 conversations, 2026-06-10):

    ~/github/cmri/papers/embedding-dynamics/experiments/motionlib.py

Usage (from the embflow repo root):

    python tests/fixtures/generate_motionlib_fixture.py [motionlib-dir]

The fixture pins embflow's ports (turning_cosines, tortuosity,
speed_autocorr, adaptive_alpha, motion_signature, trajectory,
role_slot_shuffle) to motionlib's outputs on a deterministic input.
Inputs are regenerated in the tests from the same seed; only expected
OUTPUTS are stored here.
"""
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path.home() / "github/cmri/papers/embedding-dynamics/experiments"
SEED, N, D = 7, 40, 16


def make_input():
    rng = np.random.default_rng(SEED)
    E = rng.standard_normal((N, D))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def main():
    mdir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    sys.path.insert(0, str(mdir))
    import motionlib as ml

    E = make_input()
    labels = np.array(["user", "assistant"] * (N // 2))
    rng = np.random.default_rng(0)
    fixture = {
        "seed": SEED, "n": N, "d": D,
        "motion_scalars": ml.motion_scalars(E, w=8, with_alpha=True),
        "turning_cosines_first5": ml.turning_cosines(E)[:5].tolist(),
        "tortuosity_w8": float(ml.tortuosity(E, 8)),
        "speeds_first5": ml.speeds(E)[:5].tolist(),
        "lag1_autocorr_speeds": float(ml.lag1_autocorr(ml.speeds(E))),
        "adaptive_alpha": float(ml.adaptive_alpha(E)),
        "smoothed_trajectory_alpha085_last": ml.smoothed_trajectory(E, 0.85)[-1].tolist(),
        "role_slot_shuffle_seed0_first2": ml.role_slot_shuffle(E, labels, rng)[:2].tolist(),
    }
    out = Path(__file__).parent / "motionlib_fixture.json"
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
