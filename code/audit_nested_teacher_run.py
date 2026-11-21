import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from aggregate_nested_dual_blind_oof import load_prediction
from build_nested_dual_blind_teacher_splits import KEY_COLUMNS, normalize, sha256_file
from prepare_nested_teacher_config import (
    expected_teacher_config,
    load_yaml,
    validate_manifest,
    validate_protocol_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fail-fast audit for one completed nested dual-blind teacher."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument("--teacher-id", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template-config", required=True)
    return parser.parse_args()


def require_single_best_checkpoint(run_dir):
    checkpoints = sorted(glob.glob(str(run_dir / "best_model_epoch*.ckpt")))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one best checkpoint in {run_dir}, found {checkpoints}"
        )
    if Path(checkpoints[0]).stat().st_size == 0:
        raise ValueError(f"Best checkpoint is empty: {checkpoints[0]}")
    return checkpoints[0]


def exact_key_coverage(expected, observed, teacher_id):
    membership = expected.merge(
        observed[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True
    )
    if len(membership) != len(expected) or not (membership["_merge"] == "both").all():
        raise ValueError(
            f"Teacher {teacher_id} predictions do not exactly cover its target block"
        )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    template_path = Path(args.template_config)
    manifest_path = Path(args.manifest)

    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    validate_manifest(manifest, args.outer_set_id, args.teacher_id)
    if sha256_file(Path(manifest["outer_train_path"])) != manifest["outer_train_sha256"]:
        raise ValueError("Outer training CSV changed after split construction")

    config = load_yaml(config_path)
    template = load_yaml(template_path)
    validate_protocol_config(config)
    expected_config = expected_teacher_config(
        template, args.outer_set_id, args.teacher_id
    )
    if config != expected_config:
        raise ValueError(
            "Teacher config differs from the Fold4 protocol template beyond "
            "experiment.name and data.dataset_name"
        )

    teacher = next(
        item
        for item in manifest["teachers"]
        if int(item["teacher_id"]) == args.teacher_id
    )
    dataset_dir = Path(teacher["dataset_dir"])
    train = normalize(
        pd.read_csv(dataset_dir / "blind_start" / "train_set0.csv", index_col=0)
    )
    val = normalize(
        pd.read_csv(dataset_dir / "blind_start" / "val_set0.csv", index_col=0)
    )
    target = normalize(
        pd.read_csv(dataset_dir / "blind_start" / "test_set0.csv", index_col=0)
    )

    prediction_path = run_dir / "test_predictions.csv"
    metrics_path = run_dir / "version_0" / "metrics.csv"
    if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing test predictions: {prediction_path}")
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    checkpoint = require_single_best_checkpoint(run_dir)

    raw = pd.read_csv(prediction_path)
    required_columns = set(KEY_COLUMNS) | {"logit", "prob"}
    missing_columns = sorted(required_columns - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Prediction CSV is missing columns: {missing_columns}")
    if len(raw) != int(teacher["target_rows"]):
        raise ValueError(
            f"Prediction row count {len(raw)} != target_rows {teacher['target_rows']}"
        )
    if not np.isfinite(raw["logit"].astype(float)).all():
        raise ValueError("Exported logits are non-finite")
    if not np.isfinite(raw["prob"].astype(float)).all():
        raise ValueError("Exported probabilities are non-finite")

    prediction = load_prediction(prediction_path, args.teacher_id)
    exact_key_coverage(target, prediction, args.teacher_id)
    if prediction.duplicated(KEY_COLUMNS).any():
        raise ValueError("Prediction keys are duplicated")

    seen_drugs = set(train["head"]) | set(val["head"])
    seen_proteins = set(train["tail"]) | set(val["tail"])
    if set(prediction["head"]) & seen_drugs:
        raise ValueError("Held-out target drugs leak into teacher train/validation")
    if set(prediction["tail"]) & seen_proteins:
        raise ValueError("Held-out target proteins leak into teacher train/validation")

    labels = raw["label"].astype(int).to_numpy()
    logits = raw["logit"].astype(float).to_numpy()
    probability = prediction["oof_base_prob"].astype(float).to_numpy()
    precision, recall, _ = precision_recall_curve(labels, logits)
    summary = {
        "teacher_id": args.teacher_id,
        "rows": len(raw),
        "positive_rows": int(labels.sum()),
        "target_auc": float(roc_auc_score(labels, logits)),
        "target_aupr": float(auc(recall, precision)),
        "score_source": sorted(prediction["oof_score_source"].unique().tolist()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "checkpoint": checkpoint,
        "outer_val_accessed": False,
        "outer_test_accessed": False,
    }
    print(json.dumps(summary, indent=2))
    print(f"TEACHER{args.teacher_id:02d} OOF AUDIT OK")


if __name__ == "__main__":
    main()
