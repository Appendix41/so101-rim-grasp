import unittest

import numpy as np

from cloth_task import DEV, GAP, LAYOUTS, SIDE, TRAIN, scene_for, shifted_scene, whole_disk_margin
from placement_scene import scene_split


class ClothSceneTest(unittest.TestCase):
    def test_fixed_geometry_reachable_fixture_and_split(self):
        for split, seeds in (("train", TRAIN), ("validation", DEV)):
            for s in seeds:
                spec = scene_for(s)
                self.assertEqual(scene_split(s), split)
                self.assertEqual((spec.diameter_mm, spec.height_mm), (100,90))
                centers = spec.to_world(spec.box_centers_local)
                self.assertAlmostEqual(centers[1,0]-centers[0,0]-SIDE,GAP)
                self.assertGreater(spec.start_xy[0]-.05,centers[1,0]+SIDE/2)

    def test_disk_margin_and_layout_reversal(self):
        spec = scene_for(TRAIN[0])
        xy = spec.to_world(spec.box_centers_local)
        for layout in LAYOUTS:
            i=layout.index("1")
            np.testing.assert_allclose(whole_disk_margin(spec,xy[i],i), .025)
            self.assertLess(whole_disk_margin(spec,xy[i]+[.026,0],i),0)
            self.assertLess(whole_disk_margin(spec,xy[1-i],i),0)

    def test_shifted_geometries_obey_rotated_edges(self):
        centers=[]
        for s in TRAIN:
            a=shifted_scene(s)
            self.assertEqual(a,shifted_scene(s))
            xy=a.to_world(a.box_centers_local)
            centers.append(xy)
            self.assertLess(np.linalg.norm(xy[1]-a.start_xy),np.linalg.norm(xy[0]-a.start_xy))
            self.assertGreater(a.box_centers_local[1][0]-a.box_centers_local[0][0],SIDE+.0119)
            for j in range(2):
                pt=np.asarray(a.box_centers_local[j])+[0,.026]
                self.assertAlmostEqual(whole_disk_margin(a,a.to_world(pt),j),-.001)
                self.assertAlmostEqual(whole_disk_margin(a,xy[j],j),.025)
        self.assertGreater(np.ptp(centers,axis=0).min(),.002)


if __name__ == "__main__":
    unittest.main()
