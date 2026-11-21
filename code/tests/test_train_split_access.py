import unittest
from unittest.mock import patch

import pandas as pd

from train import load_dataframes


class TrainSplitAccessTests(unittest.TestCase):
    def test_development_run_does_not_open_test_split(self):
        frame = pd.DataFrame({"head": [1], "tail": [2], "label": [1]})
        config = {
            "data": {
                "dataset_name": "example",
                "base_path": ".",
                "set_type": "blind_start",
                "set_id": 4,
                "train_fraction": 1.0,
                "val_fraction": 1.0,
            },
            "trainer": {"seed": 1234},
        }

        requested = []

        def fake_load_split_df(_config, split):
            requested.append(split)
            if split == "test":
                raise AssertionError("test split must not be accessed")
            return frame.copy()

        with patch("train.load_split_df", side_effect=fake_load_split_df):
            train, val, test = load_dataframes(config, load_test=False)

        self.assertEqual(requested, ["train", "val"])
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertIsNone(test)


if __name__ == "__main__":
    unittest.main()
