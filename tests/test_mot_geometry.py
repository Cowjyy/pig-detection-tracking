import unittest

from tools.mot_geometry import clip_small_overflow, parse_gt, validate_gt


class MotGeometryTests(unittest.TestCase):
    def test_small_overflow_preserves_id_and_clips_to_bottom(self):
        row = [1, 7, 738.12, 1021.07, 311.35, 63.11, 1, 1, 1]
        updated, changed = clip_small_overflow(row, 1920, 1080)
        self.assertTrue(changed)
        self.assertEqual(updated[:4], row[:4])
        self.assertEqual(updated[4], row[4])
        self.assertAlmostEqual(updated[3]+updated[5], 1080)
        self.assertEqual(updated[6:], row[6:])
        self.assertEqual(row[5], 63.11)  # Input is never mutated.

    def test_large_overflow_or_excessive_area_loss_requires_review(self):
        for row in ([1, 1, 0, 1070, 100, 50, 1, 1, 1], [1, 1, 0, 1079, 100, 5, 1, 1, 1]):
            with self.assertRaises(ValueError):
                clip_small_overflow(row, 1920, 1080)

    def test_zero_width_not_filled_in(self):
        with self.assertRaises(ValueError):
            clip_small_overflow([1, 6, 155, 201, 0, 3, 1, 1, 1], 1920, 1080)

    def test_duplicate_and_out_of_range_frames_fail(self):
        row = '1,1,10,10,50,50,1,1,1'
        with self.assertRaises(ValueError):
            validate_gt([row, row], 100, 100, 10)
        with self.assertRaises(ValueError):
            validate_gt(['11,1,10,10,50,50,1,1,1'], 100, 100, 10)

    def test_nonfinite_or_noninteger_ids_fail(self):
        for text in ('1,1,0,0,nan,10,1,1,1', '1,1.5,0,0,10,10,1,1,1'):
            with self.assertRaises(ValueError):
                parse_gt(text)


if __name__ == '__main__':
    unittest.main()
