"""Seeded scene specifications shared by collection, conversion, and tests."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementScene:
    seed: int
    height_mm: int
    diameter_mm: int
    start_xy: tuple[float, float]
    paper_xy: tuple[float, float]
    paper_yaw_deg: float
    box_centers_local: tuple[tuple[float, float], tuple[float, float]]
    box_sides_mm: tuple[float, float]
    light_intensity: float
    overhead_eye: tuple[float, float, float]

    @property
    def rotation(self):
        yaw = np.deg2rad(self.paper_yaw_deg)
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def to_world(self, local_xy):
        return np.asarray(local_xy) @ self.rotation.T + self.paper_xy

    def to_paper(self, world_xy):
        return (np.asarray(world_xy) - self.paper_xy) @ self.rotation

    def as_dict(self):
        return asdict(self)


def scene_split(seed: int) -> str:
    """Stable grouping independent of input order, source files, or episode count."""
    bucket = int.from_bytes(
        hashlib.sha256(f"so101-placement-v2:{seed}".encode()).digest()[:8], "big"
    ) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def sample_scene(
    seed: int,
    *,
    randomized: bool,
    heights=(90,),
    diameters=(100,),
    xy_jitter_mm=3.0,
) -> PlacementScene:
    rng = np.random.default_rng(seed)
    height = int(rng.choice(heights)) if randomized else int(heights[0])
    diameter = int(rng.choice(diameters)) if randomized else int(diameters[0])
    start_xy = np.array([0.080, -0.265]) + rng.uniform(
        -xy_jitter_mm, xy_jitter_mm, size=2
    ) / 1000
    paper_xy = np.array([-0.13, -0.265])
    centers = np.array([[-0.070, 0.0], [0.070, 0.0]])
    sides = np.array([120.0, 120.0])
    yaw, intensity = 0.0, 1700.0
    eye = np.array([0.12, -0.58, 0.58])
    if randomized:
        paper_xy += rng.uniform(-0.006, 0.006, size=2)
        yaw = float(rng.uniform(-3.0, 3.0))
        centers += rng.uniform(-0.003, 0.003, size=(2, 2))
        sides = rng.uniform(max(112.0, diameter + 18.0), 126.0, size=2)
        intensity = float(rng.uniform(1450.0, 1950.0))
        eye += rng.uniform(-0.008, 0.008, size=3)
    scene = PlacementScene(
        seed=int(seed),
        height_mm=height,
        diameter_mm=diameter,
        start_xy=tuple(start_xy),
        paper_xy=tuple(paper_xy),
        paper_yaw_deg=yaw,
        box_centers_local=tuple(tuple(xy) for xy in centers),
        box_sides_mm=tuple(sides),
        light_intensity=intensity,
        overhead_eye=tuple(eye),
    )
    validate_scene(scene)
    return scene


def validate_scene(scene: PlacementScene):
    centers = np.asarray(scene.box_centers_local)
    half_sides = np.asarray(scene.box_sides_mm) / 2000
    if not np.all(np.abs(centers) + half_sides[:, None] + 0.002 < [0.1485, 0.105]):
        raise ValueError("Target frames extend beyond the A4 sheet")
    if not np.any(np.abs(centers[0] - centers[1]) > half_sides.sum() + 0.004):
        raise ValueError("Target frames overlap")
    if np.any(np.asarray(scene.box_sides_mm) < scene.diameter_mm + 12):
        raise ValueError("Insufficient target clearance")


def adaptive_grasp_z(height_mm: int) -> float:
    return min(0.073, max(0.043, height_mm / 1000.0 - 0.017))
