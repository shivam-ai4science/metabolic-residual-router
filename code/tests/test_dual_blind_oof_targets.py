import unittest

import pandas as pd

from build_dual_blind_oof_targets import aggregate_teachers, assert_dual_entity_blind


class DualBlindOOFTargetTests(unittest.TestCase):
    def test_aggregates_multiple_eligible_teachers(self):
        first = pd.DataFrame(
            [
                {
                    "head": "1",
                    "tail": "10",
                    "label": 1,
                    "teacher_probability": 0.2,
                    "teacher_fold": 0,
                }
            ]
        )
        second = pd.DataFrame(
            [
                {
                    "head": "1",
                    "tail": "10",
                    "label": 1,
                    "teacher_probability": 0.8,
                    "teacher_fold": 2,
                }
            ]
        )
        result = aggregate_teachers([first, second]).iloc[0]
        self.assertAlmostEqual(result["oof_base_prob"], 0.5)
        self.assertAlmostEqual(result["oof_teacher_std"], 0.3)
        self.assertEqual(result["oof_teacher_count"], 2)
        self.assertEqual(result["oof_teacher_folds"], "0,2")

    def test_rejects_teacher_prediction_with_either_entity_seen(self):
        teacher_train = pd.DataFrame(
            [{"head": "1", "tail": "10", "label": 1}]
        )
        predictions = pd.DataFrame(
            [
                {"head": "1", "tail": "20", "label": 0},
                {"head": "2", "tail": "10", "label": 0},
            ]
        )
        with self.assertRaisesRegex(ValueError, "not dual-entity blind"):
            assert_dual_entity_blind(predictions, teacher_train, fold=0)


if __name__ == "__main__":
    unittest.main()
