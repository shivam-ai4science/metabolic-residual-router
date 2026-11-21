import unittest

import numpy as np
import pandas as pd

from aggregate_nested_dual_blind_oof import attach_prediction_scores, stable_sigmoid


class NestedOOFAggregationTests(unittest.TestCase):
    def test_exported_logits_preserve_order_hidden_by_saturated_probabilities(self):
        frame = pd.DataFrame(
            {
                "head": ["1", "2", "3", "4"],
                "tail": ["10", "20", "30", "40"],
                "label": [0, 0, 1, 1],
            }
        )
        raw = frame.copy()
        raw["logit"] = [-80.0, -40.0, 40.0, 80.0]
        raw["prob"] = [0.0, 0.0, 1.0, 1.0]

        scored = attach_prediction_scores(frame.copy(), raw, teacher_id=0)

        self.assertEqual(scored["oof_score_source"].unique().tolist(), ["exported_logit"])
        np.testing.assert_array_equal(scored["oof_base_logit"], raw["logit"])
        self.assertLess(scored.loc[0, "oof_base_prob"], scored.loc[1, "oof_base_prob"])
        self.assertLessEqual(scored.loc[2, "oof_base_prob"], scored.loc[3, "oof_base_prob"])
        np.testing.assert_array_equal(scored["oof_base_prob_reported"], raw["prob"])

    def test_probability_fallback_remains_supported(self):
        frame = pd.DataFrame(
            {"head": ["1", "2"], "tail": ["10", "20"], "label": [0, 1]}
        )
        raw = frame.copy()
        raw["prob"] = [0.25, 0.75]

        scored = attach_prediction_scores(frame.copy(), raw, teacher_id=1)

        np.testing.assert_allclose(scored["oof_base_prob"], [0.25, 0.75])
        np.testing.assert_allclose(stable_sigmoid(scored["oof_base_logit"]), [0.25, 0.75])
        self.assertEqual(scored["oof_score_source"].unique().tolist(), ["prob_fallback"])


if __name__ == "__main__":
    unittest.main()
