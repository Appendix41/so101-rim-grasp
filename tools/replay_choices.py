"""Replay frozen placement selectors on development RGB; score via existing simulator branches."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cloth_goal import consistent_goal
from cloth_landing_wm import choose, load
from geometric_rule import geometric_rule
from visual_proposals import anonymous_regions, visual_label


def same(x, y):
    if x is None or y is None:
        return x is y
    return np.allclose(x, y, atol=1e-6, rtol=0)


def replay():
    report = json.loads((ROOT / "evidence/dev-b.json").read_text())
    protocol = json.loads((ROOT / "evidence/dev-b-protocol.json").read_text())
    wm = load(ROOT / "models/landing", protocol["landing_sha256"])
    settings = protocol["world_model"]
    totals = {kind: {arm: [] for arm in ("raw", "wm", "rule")} for kind in ("natural", "probe")}
    with np.load(ROOT / "data/placement-dev-b.npz", allow_pickle=False) as data:
        for i, case in enumerate(report["cases"]):
            assert (data["seed"][i], data["layout"][i]) == (case["scene"], case["layout"])
            imgs, q = data["overhead_rgb"][i], data["joint_pos"][i, -1]
            cal = [data[k][i] for k in ("eye", "quat", "K")]
            index, xys = consistent_goal(data["pixels"][i], *cal)
            student = None if index is None else xys[index]
            regions = anonymous_regions(imgs, cal)
            expert, inset, _ = visual_label(imgs, cal, student)
            menu = list(data["menu"][i])
            for t in case["trials"]:
                injected = t["kind"] != "natural"
                if injected:
                    j = t["raw"]["branch"]
                    prop, alts = menu[j], menu[:j] + menu[j + 1:]
                else:
                    prop, alts = student, regions
                goal, _ = choose(wm, imgs[-1], q, cal, prop, alts, settings)
                rule, _ = geometric_rule(prop, alts, expert, inset, injected)
                for arm, xy in (("raw", prop), ("wm", goal), ("rule", rule)):
                    # Physical outcomes are joined only *after* selection.
                    expected = case["executions"][t[arm]["branch"]]
                    assert same(xy, expected["requested_xy"]), (case["scene"], t["kind"], arm)
                    totals["probe" if injected else "natural"][arm].append(expected["complete"])
    for kind, arms in totals.items():
        print(f"{kind}: " + ", ".join(f"{name} {sum(flags)}/{len(flags)}" for name, flags in arms.items()))
    return totals


if __name__ == "__main__":
    replay()
