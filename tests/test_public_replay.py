import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit import main as audit
from refit_landing import refit
from replay_choices import replay
from cloth_lift_wm import load_frozen
from placement_scene import scene_split


class ReleaseRegression(unittest.TestCase):
    def test_physical_counts(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            audit()
        scores = {" ".join(line.split()[:-1]): line.split()[-1] for line in text.getvalue().splitlines()}
        self.assertEqual(scores["student placement"], "14/16")
        self.assertEqual(scores["with landing model"], "16/16")

    def test_original_banks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(refit()["X"].shape, (39, 19))
        self.assertEqual(len(load_frozen(ROOT / "models/grasp")["trajectories"]), 32)

    def test_decision_replay(self):
        with contextlib.redirect_stdout(io.StringIO()):
            pools = replay()
        for kind, scores in (("natural", [7, 8, 8]), ("probe", [8, 28, 32])):
            self.assertEqual([sum(pools[kind][key]) for key in ("raw", "wm", "rule")], scores)

    def test_observation_contract(self):
        with np.load(ROOT / "data/placement-train.npz", allow_pickle=False) as train:
            self.assertTrue(all(scene_split(int(s)) == "train" for s in train["seed"]))
            for k in ("wrist_rgb", "overhead_rgb"):
                self.assertEqual(train[k].shape, (27, 2, 3, 240, 320, 3))
                self.assertEqual(train[k].dtype, np.uint8)
            self.assertEqual(train["uv"].shape, (27, 2, 2))
            np.testing.assert_array_equal(train["weight"][-8:], [3, 2, 3, 2, 3, 2, 3, 1])
        with np.load(ROOT / "data/placement-dev-b.npz", allow_pickle=False) as dev:
            self.assertTrue(all(scene_split(int(s)) == "validation" for s in dev["seed"]))
            self.assertNotIn("uv", dev.files)
            self.assertEqual(dev["pixels"].shape, (8, 3, 2))


if __name__ == "__main__":
    unittest.main()
