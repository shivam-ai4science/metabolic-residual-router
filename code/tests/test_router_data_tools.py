import unittest

import pandas as pd

from build_router_crossfit_targets import combine_prediction_frames
from make_bindingdb_router_crossfit_splits import assign_groups


class RouterDataToolTests(unittest.TestCase):
    def test_oof_prediction_files_must_cover_unique_pairs(self):
        first = pd.DataFrame(
            [{"head": "Drug:1", "tail": "Protein:10", "label": 1, "prob": 0.8}]
        )
        second = pd.DataFrame(
            [{"head": "2", "tail": "20", "label": 0, "prob": 0.2}]
        )
        predictions = combine_prediction_frames([first, second])
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions["teacher_fold"].tolist(), [0, 1])
        self.assertEqual(predictions["head"].tolist(), ["1", "2"])

    def test_group_assignment_has_no_entity_split(self):
        frame = pd.DataFrame(
            {
                "head": [str(index % 4) for index in range(60)],
                "tail": [str(index % 12) for index in range(60)],
                "label": [index % 2 for index in range(60)],
            }
        )
        assignments = assign_groups(frame, "tail", num_folds=3, seed=1234)
        self.assertEqual(set(assignments), set(frame["tail"]))
        self.assertEqual(set(assignments.values()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
