import unittest

import pandas as pd

from build_nested_dual_blind_teacher_splits import build_teacher_frames


class NestedDualBlindTeacherSplitTests(unittest.TestCase):
    def make_frame(self):
        rows = []
        for drug in range(9):
            for protein in range(9):
                rows.append(
                    {
                        "head": str(drug),
                        "tail": str(protein),
                        "label": (drug + protein) % 2,
                    }
                )
        return pd.DataFrame(rows)

    def test_every_row_is_predicted_once_and_both_entities_are_hidden(self):
        source = self.make_frame()
        teachers, _, _, _, _ = build_teacher_frames(
            source,
            num_buckets=3,
            val_frac=0.2,
            seed=1234,
        )
        self.assertEqual(len(teachers), 9)
        targets = pd.concat([teacher["target"] for teacher in teachers], ignore_index=True)
        self.assertEqual(len(targets), len(source))
        self.assertFalse(targets.duplicated(["head", "tail", "label"]).any())

        for teacher in teachers:
            seen = pd.concat([teacher["train"], teacher["val"]], ignore_index=True)
            self.assertFalse(set(teacher["target"]["head"]) & set(seen["head"]))
            self.assertFalse(set(teacher["target"]["tail"]) & set(seen["tail"]))


if __name__ == "__main__":
    unittest.main()
