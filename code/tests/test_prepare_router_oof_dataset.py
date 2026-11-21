import unittest

import pandas as pd

from prepare_router_oof_dataset import parse_teacher_folds, validate_oof_train


class PrepareRouterOOFDatasetTests(unittest.TestCase):
    def make_frame(self):
        return pd.DataFrame(
            {
                "head": [1, 2],
                "tail": [10, 20],
                "label": [1, 0],
                "oof_base_prob": [0.8, 0.2],
                "oof_base_logit": [1.386294, -1.386294],
                "oof_teacher_count": [2, 1],
                "oof_teacher_std": [0.1, 0.0],
                "oof_teacher_folds": ["0,2", "1"],
            }
        )

    def test_accepts_aggregated_multi_teacher_schema(self):
        teacher_sets = validate_oof_train(self.make_frame())
        self.assertEqual(teacher_sets.tolist(), [[0, 2], [1]])

    def test_rejects_teacher_count_mismatch(self):
        frame = self.make_frame()
        frame.loc[0, "oof_teacher_count"] = 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_oof_train(frame)

    def test_teacher_folds_are_deduplicated(self):
        self.assertEqual(parse_teacher_folds("2,0,2"), [0, 2])


if __name__ == "__main__":
    unittest.main()
