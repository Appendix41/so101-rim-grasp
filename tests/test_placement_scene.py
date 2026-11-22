import unittest

import numpy as np

from placement_scene import adaptive_grasp_z, sample_scene, scene_split


class SceneTests(unittest.TestCase):
    def test_seed_reproducibility_and_diversity(self):
        kwargs = dict(
            randomized=True, heights=(60, 90, 120), diameters=(90, 100), xy_jitter_mm=12
        )
        scenes = [sample_scene(seed, **kwargs) for seed in range(200)]
        self.assertEqual(scenes[0], sample_scene(0, **kwargs))
        self.assertEqual({s.height_mm for s in scenes}, {60, 90, 120})
        self.assertEqual({s.diameter_mm for s in scenes}, {90, 100})
        self.assertEqual(len({s.paper_xy for s in scenes}), 200)
        self.assertEqual(len({s.box_sides_mm for s in scenes}), 200)

    def test_coordinates_round_trip(self):
        scene = sample_scene(9, randomized=True)
        points = np.array(scene.box_centers_local)
        np.testing.assert_allclose(scene.to_paper(scene.to_world(points)), points)

    def test_fixed_scene_and_height_offset(self):
        scene = sample_scene(9, randomized=False)
        self.assertEqual(scene.box_sides_mm, (120.0, 120.0))
        self.assertAlmostEqual(adaptive_grasp_z(60), 0.043)
        self.assertAlmostEqual(adaptive_grasp_z(90), 0.073)
        self.assertAlmostEqual(adaptive_grasp_z(120), 0.073)

    def test_group_split_is_order_independent(self):
        seeds = list(range(100))
        expected = {seed: scene_split(seed) for seed in seeds}
        self.assertEqual(expected, {seed: scene_split(seed) for seed in reversed(seeds)})
        self.assertEqual(set(expected.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
