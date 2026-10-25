"""Checks for conversion semantics, grouped splits and raw-directory write guards."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_detection import convert_box
from scripts.make_splits import assign_groups
from tools.preparation import new_output, source_path


class PreparationTests(unittest.TestCase):
    def test_voc_inclusive_preserves_border_pixels(self):
        self.assertEqual(convert_box([1, 1, 1280, 720], 1280, 720, "voc_1based_inclusive"), [0, 0, 1280, 720])

    def test_half_open_does_not_add_a_pixel(self):
        self.assertEqual(convert_box([0, 0, 1280, 720], 1280, 720, "zero_based_half_open"), [0, 0, 1280, 720])

    def test_zero_minimum_is_not_silently_treated_as_voc(self):
        with self.assertRaises(ValueError):
            convert_box([0, 1, 10, 10], 20, 20, "voc_1based_inclusive")

    def test_invalid_box_is_not_clamped(self):
        for box in ([5, 5, 5, 6], [-1, 0, 10, 10], [0, 0, 21, 10]):
            with self.assertRaises(ValueError):
                convert_box(box, 20, 20, "zero_based_half_open")

    def test_groups_are_disjoint_complete_and_reproducible(self):
        groups = {f"scene_{i}": [f"{i}_{j}" for j in range(i+1)] for i in range(12)}
        first, counts = assign_groups(groups, [0.7, 0.2, 0.1], 42)
        second, _ = assign_groups(dict(reversed(list(groups.items()))), [0.7, 0.2, 0.1], 42)
        self.assertEqual(first, second)
        assigned = [group for values in first.values() for group in values]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(groups))
        self.assertTrue(all(first.values()))
        self.assertEqual(sum(counts.values()), sum(map(len, groups.values())))

    def test_too_few_independent_groups_fails(self):
        with self.assertRaises(ValueError):
            assign_groups({"same_video": ["1", "2", "3"]}, [0.7, 0.2, 0.1], 42)

    def test_output_cannot_enter_raw_or_escape_allowed_directories(self):
        for path in ("pig-objection detection/new", "PIG_MOT_TEST_DATA/new", "configs/new", "datasets/processed/../../PIG_MOT_TEST_DATA/new"):
            with self.assertRaises(ValueError):
                new_output(path)

    def test_source_must_stay_in_raw_dataset(self):
        with self.assertRaises(ValueError):
            source_path("README.md")


if __name__ == "__main__":
    unittest.main()
