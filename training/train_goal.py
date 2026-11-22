"""Train a pixel-goal LoRA on the public train pairs. Never emits robot commands."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cloth_task import TASK
from placement_scene import scene_split


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Config, PI05Policy
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.data import default_collate
    from cloth_goal import goal_readout

    if not torch.cuda.is_available():
        raise SystemExit("Pixel-head training requires CUDA; tools/audit.py and tools/replay_choices.py do not.")
    if not args.base.is_dir() or not args.tokenizer.is_dir():
        raise SystemExit("Provide local, separately obtained pi05_base and PaliGemma tokenizer directories.")
    args.out.mkdir(parents=True, exist_ok=False)
    cfg_dir = ROOT / "training/config"
    training = json.loads((ROOT / "configs/cloth-dagger-01.json").read_text())["training"]
    v1 = json.loads((ROOT / "evidence/goal-v1-update.json").read_text())
    rng_seed = training["seed"] if args.stage == "dagger" else 250001
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = PI05Config.from_pretrained(str(cfg_dir), device="cuda")
    config.gradient_checkpointing = False
    base = PI05Policy.from_pretrained(str(args.base), config=config, strict=True, local_files_only=True)
    if args.init:
        model = PeftModel.from_pretrained(base, str(args.init), is_trainable=True)
        parent = digest(args.init / "adapter_model.safetensors")
    else:
        if args.stage != "distill":
            raise ValueError("DAgger starts from a placement v1 adapter; first run --stage distill.")
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=json.loads((cfg_dir / "lora-targets.json").read_text())["target_modules"]))
        parent = "fresh-rank8"  # A runnable variant, not the historical grasp-initialized v1.
    model.to("cuda").eval()
    pre, _ = make_pre_post_processors(policy_cfg=config, pretrained_path=str(cfg_dir),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(args.tokenizer.resolve())}})
    params = []
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
        if p.requires_grad:
            params.append(p)
    assert sum(p.numel() for p in params) == 643584
    with np.load(ROOT / "data/placement-train.npz", allow_pickle=False) as f:
        data = dict(f)
    assert all(scene_split(int(s)) == "train" for s in data["seed"])
    old = np.flatnonzero(data["source"] != "dagger")
    fresh = np.flatnonzero(data["source"] == "dagger")
    assert len(old) == 19 and len(fresh) == 8
    center, scale = (np.array(v1[k], dtype=np.float32) for k in ("center", "scale"))
    total = training["steps"] if args.stage == "dagger" else 240
    steps = total if args.steps is None else args.steps
    if not 1 <= steps <= total:
        raise ValueError(f"--steps must be in [1, {total}]")
    lr = training["lr"] if args.stage == "dagger" else .00019
    opt = torch.optim.AdamW(params, lr=lr, betas=(.9, .95), weight_decay=.01)
    pool, new_pool = [], []
    with (args.out / "loss.jsonl").open("x") as log, sdpa_kernel(backends=[SDPBackend.MATH]):
        for step in range(steps):
            if args.stage == "dagger":
                if not pool:
                    pool = list(rng.permutation(old))
                if not new_pool:
                    new_pool = list(rng.permutation(np.repeat(fresh, data["weight"][fresh])))
                pair_ids = [pool.pop(), new_pool.pop()]
            else:
                while len(pool) < 2:
                    pool.extend(rng.permutation(old))
                pair_ids, pool = pool[:2], pool[2:]
            frame = int(rng.integers(3))
            obs = []
            for pair in pair_ids:
                for side in range(2):
                    raw = {"task": TASK, "observation.state": torch.from_numpy(
                        data["joint_pos"][pair, side, frame].astype(np.float32))}
                    for camera, key in (("overhead_rgb", "base_0_rgb"), ("wrist_rgb", "left_wrist_0_rgb")):
                        im = data[camera][pair, side, frame]
                        raw["observation.images." + key] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.
                    obs.append(raw)
            y = (data["uv"][pair_ids].reshape(4, 2) - center) / scale
            err = goal_readout(model.get_base_model(), pre(default_collate(obs))) - torch.from_numpy(y).cuda()
            loss = err.square().mean() + .35 * (err[1::2] - err[::2]).square().mean()
            opt.param_groups[0]["lr"] = lr * (.15 + .85 * (1 + math.cos(math.pi * step / (total - 1))) / 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            print(json.dumps(dict(step=step + 1, loss=loss.item(), pair=[int(v) for v in pair_ids], frame=frame)), file=log, flush=True)
    model.save_pretrained(str(args.out))
    (args.out / "goal_readout.json").write_text(json.dumps(dict(mode="pixel_goal_only_not_joint_actions",
        stage=args.stage, steps=steps, parent=parent, sha256=digest(args.out / "adapter_model.safetensors"),
        task=TASK, center=center.tolist(), scale=scale.tolist()), indent=2) + "\n")
    print(f"Saved {steps} {args.stage} updates to {args.out}. This checkpoint predicts pixels, not joint commands.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--init", type=Path, help="Optional grasp LoRA for distill; required pixel-goal v1 for dagger")
    ap.add_argument("--stage", choices=("distill", "dagger"), required=True)
    ap.add_argument("--steps", type=int, help="Smoke run length; default is the full registered schedule")
    ap.add_argument("--out", type=Path, required=True, help="New local directory; never pushes to a Hub")
    main(ap.parse_args())
