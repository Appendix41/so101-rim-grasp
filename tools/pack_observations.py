"""Export a small, allowlisted slice from the original Isaac experiment directory."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cloth_goal import pixel_of
from placement_scene import scene_split


def read_json(path):
    return json.loads(path.read_text())


def three_views(branch, frames):
    return {
        key: np.stack([branch[f"{stage}/{key}"][idx] for stage, idx in frames])
        for key in ("overhead_rgb", "wrist_rgb", "joint_pos")
    }


LAST = (("observed", -8), ("observed", -4), ("terminal", ()))
EARLY = (("observed", 0), ("observed", 3), ("observed", 6))


def export(runs, out):
    old = read_json(runs / "cloth-wm-color-fit-01/inputs.json")["pairs"]
    new = read_json(runs / "cloth-dagger-fit-01/recipe.json")["new_pairs"]
    paired = []
    for row in [r for r in old if r["paired"]] + new:
        seed = row["seed"]
        assert scene_split(seed) == "train"
        kind = row.get("source", "dagger")
        folder, file = {
            "held_teacher": ("cloth-shifted-teacher-01", "held.h5"),
            "regrasp_train": ("cloth-wm-train-01", "prelift.h5"),
            "dagger": ("cloth-dagger-train-01", "prelift.h5"),
        }[kind]
        sides = []
        with h5py.File(runs / folder / file) as f:
            assert not f.attrs["hardware_accessed"]
            for j, layout in enumerate(("12", "21")):
                cell = f[f"{seed}/{layout}"]
                if kind == "held_teacher":
                    obs = three_views(cell["rgb"], EARLY)
                elif kind == "regrasp_train":
                    obs = three_views(cell["vision/branches/vision"], LAST)
                else:
                    c = cell["paired_goals"]
                    obs = three_views(c["branches/" + c.attrs["selected"]], LAST)
                    h = c["handoff"]
                    cal = [h["calibration/" + k][:] for k in ("eye", "quat", "K")]
                    uv = pixel_of(h.attrs["teacher"], *cal)
                obs["uv"] = np.asarray(uv if kind == "dagger" else row["pixels"][j], dtype=np.float32)
                sides.append(obs)
        paired.append({**{k: np.stack([s[k] for s in sides]) for k in sides[0]},
                       "seed": seed, "source": kind,
                       "weight": 1 + sum(r != "inside_safe_region" for r in row.get("reasons", []))})
    assert len(paired) == 27 and len({r["seed"] for r in paired}) == 27
    np.savez_compressed(out / "placement-train.npz", **{k: np.stack([r[k] for r in paired]) for k in paired[0]})

    # This separate development fixture contains only information available before acting.
    demo = []
    run = runs / "cloth-candidate-challenge-01"
    with h5py.File(run / "prelift.h5") as f:
        assert not f.attrs["hardware_accessed"]
        for seed in read_json(run / "protocol.json")["scenes"]:
            assert scene_split(seed) == "validation"
            for layout in ("12", "21"):
                c = f[f"{seed}/{layout}/paired_goals"]
                h = c["handoff"]
                obs = three_views(c["branches/" + c.attrs["selected"]], LAST)
                demo.append(dict(seed=seed, layout=layout, **obs, pixels=h["pixels"][:], menu=h["menu"][:],
                                 **{k: h["calibration/" + k][:] for k in ("eye", "quat", "K")}))
    np.savez_compressed(out / "placement-dev-b.npz", **{k: np.stack([r[k] for r in demo]) for k in demo[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Original simulator run archive (not needed for public replay)")
    parser.add_argument("output", type=Path, help="Existing directory, must not contain either output")
    args = parser.parse_args()
    if any((args.output / f).exists() for f in ("placement-dev-b.npz", "placement-train.npz")):
        parser.error("refusing to overwrite an export")
    export(args.runs, args.output)
