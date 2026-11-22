import unittest

import numpy as np

from cloth_lift_wm import GROUPS, HORIZON, TICKS, forecast, timed_outcome


class LiftModelTest(unittest.TestCase):
    def test_contiguous_ticks_only(self):
        path = np.zeros((TICKS,2))
        path[20:67,0] = 20
        path[68:115,0] = 20
        self.assertIsNone(timed_outcome(path))
        path[115] = (20,15)
        self.assertEqual(timed_outcome(path),116/120)
        self.assertIsNone(timed_outcome(path,True))
        path[115,1] = 15.01
        self.assertIsNone(timed_outcome(path))

    def test_action_conditioning_and_cost(self):
        x = np.zeros(sum(GROUPS))
        action = np.repeat([0,1],4)
        y = np.zeros((8,TICKS,2))
        y[action==1,100:,0] = 25
        model = dict(action=action,X=np.stack([x]*8),scale=np.ones_like(x),trajectories=y,
                     errors=np.zeros(8,bool))
        live, vision = [forecast(model,x,i) for i in (0,1)]
        self.assertAlmostEqual(live["cost"],HORIZON+12)
        self.assertAlmostEqual(vision["cost"],148/120+.35)
        self.assertAlmostEqual(vision["p_held"],1.)
        self.assertAlmostEqual(live["p_held"],0.)
        np.testing.assert_allclose(live["mean"],y[0])
        np.testing.assert_allclose(vision["mean"],y[-1])


if __name__=="__main__":
    unittest.main()
