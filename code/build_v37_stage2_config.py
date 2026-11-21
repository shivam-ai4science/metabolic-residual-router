import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Build a calibrated V3.7 Stage-2 config.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outer-set-id", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.template, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    calibration = json.loads(Path(args.calibration_summary).read_text(encoding="ascii"))
    if calibration.get("outer_val_accessed") or calibration.get("outer_test_accessed"):
        raise ValueError("Calibration summary reports outer validation/test access")
    if calibration.get("protocol") != "leave-one-teacher-block-out-platt-v1":
        raise ValueError("Unsupported calibration protocol")

    calibrator = calibration["global_calibrator"]
    fold_tag = f"fold{args.outer_set_id}"
    outer_tag = f"outer{args.outer_set_id}"
    if fold_tag not in args.source_run:
        raise ValueError(f"Source run is not tagged for {fold_tag}: {args.source_run}")
    if outer_tag not in args.dataset_name:
        raise ValueError(f"Router dataset is not tagged for {outer_tag}: {args.dataset_name}")

    config["experiment"]["name"] = (
        f"aibind_routed_residual_v37_calibrated_{fold_tag}_seed{args.seed}"
    )
    config["experiment"]["seed"] = int(args.seed)
    config["finetune"]["source_run"] = args.source_run
    config["data"]["dataset_name"] = args.dataset_name
    config["data"]["set_id"] = int(args.outer_set_id)
    config["model"]["params"]["base_logit_scale"] = float(calibrator["scale"])
    config["model"]["params"]["base_logit_bias"] = float(calibrator["bias"])
    config["trainer"]["seed"] = int(args.seed)
    config.setdefault("evaluation", {})["test_after_fit"] = False
    config["calibration"] = {
        "protocol": calibration["protocol"],
        "outer_set_id": int(args.outer_set_id),
        "summary": str(Path(args.calibration_summary).resolve()),
        "input_sha256": calibration["input_sha256"],
        "output_sha256": calibration["output_sha256"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Saved: {output_path.resolve()}")
    print(f"base_logit_scale={calibrator['scale']:.12f}")
    print(f"base_logit_bias={calibrator['bias']:.12f}")
    print("test_after_fit=false")


if __name__ == "__main__":
    main()
