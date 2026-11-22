"""Black-cloth curriculum v1. Internal slot codes 1/2 denote black/red, never digits on screen."""
from dataclasses import replace

import numpy as np

from placement_scene import sample_scene


TASK = "Put the black open-top can on the black cloth."
TRAIN = (246000, 246001, 246002, 246003, 246004, 246006)
DEV = (246005, 246013)
LAYOUTS = ("12", "21")
SIDE = .150
GAP = .025
COLORS = {"1": (.009, .009, .009), "2": (.75, .012, .015)}


def scene_for(seed):
    s = sample_scene(seed, randomized=True, heights=(90,), diameters=(100,), xy_jitter_mm=3)
    offset = (SIDE + GAP) / 2
    return replace(s, start_xy=(s.start_xy[0]+.015, s.start_xy[1]),
                   paper_xy=(-.130, -.230), paper_yaw_deg=0.,
                   box_centers_local=((-offset, 0.), (offset, 0.)), box_sides_mm=(1000*SIDE,)*2)


def shifted_scene(seed):
    """Small changes to both physical cloth regions, independently of their colors."""
    scene = scene_for(seed)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9301]))
    dx, dy = rng.uniform([-.008, -.014], [.002, .014])
    yaw = float(rng.uniform(-3., 3.))
    offset = (SIDE + rng.uniform(.012, .030)) / 2
    y0, y1 = rng.uniform(-.004, .004, size=2)
    return replace(scene, paper_xy=(scene.paper_xy[0]+dx, scene.paper_xy[1]+dy), paper_yaw_deg=yaw,
                   box_centers_local=((-offset, y0), (offset, y1)))


def whole_disk_margin(scene, poses_xy, side):
    local = scene.to_paper(poses_xy)
    return SIDE/2-scene.diameter_mm/2000-np.abs(local-scene.box_centers_local[side]).max(axis=-1)
