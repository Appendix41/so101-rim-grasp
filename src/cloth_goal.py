"""A placement-only readout of the pi0.5 action expert, not a joint command."""
import numpy as np
from scipy.spatial.transform import Rotation

from cloth_safe_target import table_points


def pixel_of(xy, eye, quat, K):
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    ray = (np.r_[xy, .0005] - eye) @ rot
    assert ray[2] > 0
    pix = K @ ray
    return pix[:2] / pix[2]


def consistent_goal(pixels, eye, quat, K):
    """Use the same 8 mm multi-frame agreement rule as the geometric reference."""
    uv = np.asarray(pixels)
    assert uv.shape == (3, 2) and np.isfinite(uv).all()
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    valid = ((uv >= 0) & (uv < [320, 240])).all(1)
    goals = np.full((3, 2), np.nan)
    if valid.any():
        goals[valid] = table_points(uv[valid], eye, rot, K)
    for j in range(1, 3):
        if np.any(np.linalg.norm(goals[:j] - goals[j], axis=1) <= .008):
            return j, goals
    return None, goals


def goal_readout(base, batch):
    import torch
    from train_cloth_shifted import prefix

    mask, kv = prefix(base, batch)
    n = mask.shape[0]
    zero = torch.zeros(n, base.config.chunk_size, base.config.max_action_dim, device=mask.device)
    velocity = base.model.denoise_step(mask, kv, zero, torch.ones(n, device=mask.device))
    return -velocity[:, :8, :2].float().mean(1)
