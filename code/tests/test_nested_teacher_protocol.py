import copy
import unittest

from prepare_nested_teacher_config import (
    expected_teacher_config,
    validate_manifest,
    validate_protocol_config,
)


def template_config():
    return {
        "experiment": {"name": "v37_outer4_nested_teacher00", "mode": "pretrain"},
        "data": {
            "dataset_name": "BindingDB_AIBind_outer4_nested_oof/teacher_00",
            "set_type": "blind_start",
            "set_id": 0,
            "read_on_fly": False,
            "batch_size_train": 256,
            "batch_size_val": 256,
            "embeddings": "coldstartcpi",
            "padding_mode": "official_fixed",
        },
        "model": {"name": "ColdstartCPIRepro"},
        "loss": {"name": "CrossEntropyFromLogitsLoss"},
        "trainer": {
            "max_epochs": 150,
            "learning_rate": 1e-4,
            "monitor_metric": "val_score",
            "monitor_mode": "max",
            "early_stopping_patience": 30,
        },
        "evaluation": {"test_after_fit": True, "export_predictions": True},
    }


class NestedTeacherProtocolTests(unittest.TestCase):
    def test_fold0_config_only_changes_names(self):
        template = template_config()
        generated = expected_teacher_config(template, outer_set_id=0, teacher_id=3)
        expected = copy.deepcopy(template)
        expected["experiment"]["name"] = "v37_outer0_nested_teacher03"
        expected["data"]["dataset_name"] = (
            "BindingDB_AIBind_outer0_nested_oof/teacher_03"
        )
        self.assertEqual(generated, expected)
        validate_protocol_config(generated)

    def test_manifest_rejects_outer_test_access(self):
        manifest = {
            "protocol": "outer-train-only-nested-dual-entity-blind-v1",
            "outer_set_id": 0,
            "outer_val_accessed": False,
            "outer_test_accessed": True,
            "teachers": [{"teacher_id": 3}],
        }
        with self.assertRaises(ValueError):
            validate_manifest(manifest, outer_set_id=0, teacher_id=3)

    def test_protocol_drift_is_rejected(self):
        config = template_config()
        config["trainer"]["early_stopping_patience"] = 10
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_yaml_style_scientific_notation_is_accepted(self):
        config = template_config()
        config["trainer"]["learning_rate"] = "1e-4"
        validate_protocol_config(config)


if __name__ == "__main__":
    unittest.main()
