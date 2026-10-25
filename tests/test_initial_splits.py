import unittest

from scripts.create_initial_splits import ordered_split


class InitialSplitTests(unittest.TestCase):
    def test_numeric_order_full_coverage_and_expected_counts(self):
        records = [{"id": str(i)} for i in range(1, 1017) if i != 558]
        split = ordered_split(list(reversed(records)))
        self.assertEqual({name: len(rows) for name, rows in split.items()}, {"train": 711, "val": 203, "test": 101})
        self.assertEqual([int(r["id"]) for rows in split.values() for r in rows], [int(r["id"]) for r in records])
        self.assertLess(int(split["train"][-1]["id"]), int(split["val"][0]["id"]))
        self.assertLess(int(split["val"][-1]["id"]), int(split["test"][0]["id"]))

    def test_duplicate_numeric_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            ordered_split([{"id": "1"}, {"id": "01"}, {"id": "2"}])

    def test_unknown_nonnumeric_order_is_rejected(self):
        with self.assertRaises(ValueError):
            ordered_split([{"id": "a"}, {"id": "2"}, {"id": "3"}])


if __name__ == "__main__":
    unittest.main()
