import unittest

import numpy as np

from scripts.check_content_overlap import comparison, qualifies


class OverlapMetricTests(unittest.TestCase):
    def test_identical_pixels_pass_comparison(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(90, 160), dtype=np.uint8)
        metrics = comparison(image, image, [], 160, 90)
        self.assertTrue(qualifies(metrics))
        self.assertAlmostEqual(metrics['ssim'], 1.0)

    def test_matching_background_does_not_hide_changed_object_roi(self):
        a = np.full((90, 160), 70, dtype=np.uint8)
        b = a.copy()
        a[30:38, 40:48] = 220
        b[30:38, 60:68] = 220
        metrics = comparison(a, b, [{'bbox_xyxy':[40,30,48,38]}], 160, 90)
        self.assertLess(metrics['mae'], 3)
        self.assertGreater(metrics['object_roi_mae'], 4)
        self.assertFalse(qualifies(metrics))


if __name__=='__main__':
    unittest.main()
