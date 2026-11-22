"""Recount physical outcomes (standard library only); no saved summary is taken on trust."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence"


def read(name):
    return json.loads((E / f"{name}.json").read_text())


def rate(flags):
    return f"{sum(flags)}/{len(flags)}"


def checked(r):
    good = all(r[k] for k in ("held", "sustained", "stable", "open", "clear", "inside")) and not r["failure"]
    assert r["complete"] == bool(good)
    return bool(good)


def main():
    for group in ("dev-a", "dev-b"):
        protocol = E / f"{group}-protocol.json"
        assert hashlib.sha256(protocol.read_bytes()).hexdigest() == read(group)["protocol_sha256"]
    a = read("dev-a")["rows"]
    b = read("dev-b")["cases"]
    train = read("dagger-recipe")
    sets = [{r["scene"] for r in a}, {c["scene"] for c in b}, {r["seed"] for r in train["new_pairs"]},
            {r["seed"] for r in read("old-train-pairs")["pairs"] if r["paired"]}]
    assert [len(s) for s in sets] == [4, 4, 8, 19]
    assert len(set.union(*sets)) == sum(map(len, sets))
    decisions = {}
    for method in ("previous", "dagger", "landing_wm", "visual_rule", "expert"):
        rows = [r for r in a if r["method"] == method]
        assert len({(r["scene"], r["layout"]) for r in rows}) == len(rows) == 8
        decisions["A " + method] = [checked(r) for r in rows]
    assert len({(c["scene"], c["layout"]) for c in b}) == len(b) == 8
    for kind in ("natural", "controlled"):
        for arm in ("raw", "wm", "rule"):
            flags = []
            for case in b:
                assert Counter(t["kind"] for t in case["trials"]) == {
                    k: 1 for k in ("natural", "red_center", "red_edge", "black_center", "black_edge")}
                for t in case["trials"]:
                    if (t["kind"] == "natural") != (kind == "natural"):
                        continue
                    row = case["executions"][t[arm]["branch"]]
                    assert all(row[k] == v for k, v in t[arm].items())
                    flags.append(checked(row))
            decisions[f"B {kind} {arm}"] = flags
    for name, flags in decisions.items():
        print(f"{name:24} {rate(flags)}")
    for name, a_arm, b_arm in (("student placement", "dagger", "raw"), ("with landing model", "landing_wm", "wm")):
        print(f"{name:24} {rate(decisions['A ' + a_arm] + decisions['B natural ' + b_arm])}")
    for arm in ("live", "vision", "selector"):
        rows = [r for r in read("grasp-ablation")["rows"] if r["condition"] == arm]
        print(f"grasp {arm:18} {rate([r['held'] for r in rows])} sustained lifts")
    for style in ("side", "rim"):
        print(f"scripted {style:15} {rate([r['label'] == 'held' for r in read(style + '-script')])} sustained lifts")


if __name__ == "__main__":
    main()
