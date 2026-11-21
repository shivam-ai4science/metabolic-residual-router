import unittest

import numpy as np
import pandas as pd

from calibrate_nested_oof_targets import calibrate_frame, crossfit_platt


class NestedOOFCalibrationTests(unittest.TestCase):
    def test_heldout_predictions_do_not_use_heldout_labels(self):
        logits = np.array([-3, -1, 1, 3, -2, -0.5, 0.5, 2, -4, -2, 2, 4], dtype=float)
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        teachers = np.repeat([0, 1, 2], 4)

        original, _ = crossfit_platt(logits, labels, teachers)
        changed_labels = labels.copy()
        changed_labels[teachers == 1] = 1 - changed_labels[teachers == 1]
        changed, _ = crossfit_platt(logits, changed_labels, teachers)

        np.testing.assert_allclose(original[teachers == 1], changed[teachers == 1])

    def test_calibrated_frame_preserves_raw_scores_and_schema(self):
        frame = pd.DataFrame(
            {
                "head": [str(i) for i in range(12)],
                "tail": [str(i + 20) for i in range(12)],
                "label": [0, 0, 1, 1] * 3,
                "oof_base_prob": [0.1, 0.2, 0.8, 0.9] * 3,
                "oof_base_logit": [-3, -1, 1, 3, -2, -0.5, 0.5, 2, -4, -2, 2, 4],
                "oof_teacher_folds": [0] * 4 + [1] * 4 + [2] * 4,
                "oof_teacher_count": [1] * 12,
                "oof_teacher_std": [0.0] * 12,
            }
        )

        calibrated, summary = calibrate_frame(frame)

        np.testing.assert_allclose(calibrated["oof_base_logit_raw"], frame["oof_base_logit"])
        self.assertTrue(calibrated["oof_base_prob"].between(0.0, 1.0).all())
        self.assertEqual(summary["teacher_count"], 3)
        self.assertFalse(summary["outer_val_accessed"])
        self.assertFalse(summary["outer_test_accessed"])


if __name__ == "__main__":
    unittest.main()
