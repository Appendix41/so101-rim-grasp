"""Rebuild the 39-sample action-conditioned model bank from TRAIN outcomes only."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from placement_scene import scene_split


def refit():
    audit = json.loads((ROOT / "evidence/train-outcomes.json").read_text())
    assert audit["split"] == "train"
    groups = defaultdict(list)
    for r in audit["rows"]:
        assert scene_split(r["scene"]) == "train"
        if r["held"] and r["requested_xy"] is not None:
            groups[(r["scene"], r["layout"], *np.round(r["requested_xy"], 6))].append(r)
    X, Y, scenes = [], [], []
    for key, rows in groups.items():
        for row in rows:
            np.testing.assert_allclose(row["X"], rows[0]["X"], atol=1e-5)
        X.append(rows[0]["X"])
        Y.append(np.mean([r["Y"] for r in rows], axis=0))
        scenes.append(key[0])
    bank = dict(X=np.array(X), Y=np.array(Y), scene=np.array(scenes))
    with np.load(ROOT / "models/landing/landing.npz", allow_pickle=False) as frozen:
        for key in bank:
            np.testing.assert_array_equal(frozen[key], bank[key])
    print(f"All three frozen arrays reproduced exactly: {len(X)} distinct actions, {len(set(scenes))} training layouts.")
    return bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional NEW npz file; default: verify only")
    args = parser.parse_args()
    bank = refit()
    if args.out:
        with args.out.open("xb") as out:
            np.savez_compressed(out, **bank)
