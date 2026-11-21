import argparse
import copy
import json
from pathlib import Path

PROTOCOL = "outer-train-only-nested-dual-entity-blind-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one nested-teacher config from the verified Fold4 template."
    )
    parser.add_argument("--template-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument("--teacher-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_yaml(path):
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected_teacher_config(template, outer_set_id, teacher_id):
    config = copy.deepcopy(template)
    teacher_tag = f"{teacher_id:02d}"
    config["experiment"]["name"] = (
        f"v37_outer{outer_set_id}_nested_teacher{teacher_tag}"
    )
    config["data"]["dataset_name"] = (
        f"BindingDB_AIBind_outer{outer_set_id}_nested_oof/teacher_{teacher_tag}"
    )
    return config


def validate_manifest(manifest, outer_set_id, teacher_id):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsafe or unsupported nested OOF protocol")
    if int(manifest.get("outer_set_id", -1)) != outer_set_id:
        raise ValueError("Manifest outer fold does not match requested outer fold")
    if manifest.get("outer_val_accessed") or manifest.get("outer_test_accessed"):
        raise ValueError("Manifest reports outer validation/test access")

    teacher_ids = {int(item["teacher_id"]) for item in manifest["teachers"]}
    if teacher_id not in teacher_ids:
        raise ValueError(f"Teacher {teacher_id} is absent from the manifest")


def validate_protocol_config(config):
    expected = {
        ("experiment", "mode"): "pretrain",
        ("data", "set_type"): "blind_start",
        ("data", "set_id"): 0,
        ("data", "read_on_fly"): False,
        ("data", "batch_size_train"): 256,
        ("data", "batch_size_val"): 256,
        ("data", "embeddings"): "coldstartcpi",
        ("data", "padding_mode"): "official_fixed",
        ("model", "name"): "ColdstartCPIRepro",
        ("loss", "name"): "CrossEntropyFromLogitsLoss",
        ("trainer", "max_epochs"): 150,
        ("trainer", "learning_rate"): 1e-4,
        ("trainer", "monitor_metric"): "val_score",
        ("trainer", "monitor_mode"): "max",
        ("trainer", "early_stopping_patience"): 30,
        ("evaluation", "test_after_fit"): True,
        ("evaluation", "export_predictions"): True,
    }
    for path, expected_value in expected.items():
        section, key = path
        actual = config.get(section, {}).get(key)
        comparable_actual = actual
        if isinstance(expected_value, float) and not isinstance(actual, bool):
            try:
                comparable_actual = float(actual)
            except (TypeError, ValueError):
                pass
        if comparable_actual != expected_value:
            raise ValueError(
                f"Protocol drift at {section}.{key}: {actual!r} != {expected_value!r}"
            )


def main():
    import yaml

    args = parse_args()
    template_path = Path(args.template_config)
    manifest_path = Path(args.manifest)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    template = load_yaml(template_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    validate_manifest(manifest, args.outer_set_id, args.teacher_id)
    validate_protocol_config(template)

    config = expected_teacher_config(template, args.outer_set_id, args.teacher_id)
    validate_protocol_config(config)

    output_path = Path(args.output)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="ascii",
    )

    reloaded = load_yaml(output_path)
    if reloaded != config:
        raise AssertionError("Generated config changed during YAML round-trip")
    print(f"CONFIG PARITY OK: {output_path}")


if __name__ == "__main__":
    main()
