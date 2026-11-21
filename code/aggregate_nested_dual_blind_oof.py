import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_nested_dual_blind_teacher_splits import (
    KEY_COLUMNS,
    filter_missing_bindingdb_features,
    normalize,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate nested dual-blind teacher predictions with leakage assertions."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="TEACHER_ID=CSV",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def parse_prediction_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Prediction spec must be TEACHER_ID=CSV, received: {spec}")
    teacher_id, path = spec.split("=", 1)
    return int(teacher_id), Path(path)


def stable_sigmoid(values):
    """Compute sigmoid in float64 without overflowing large negative logits."""
    logits = np.asarray(values, dtype=np.float64)
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    probabilities[~positive] = negative_exp / (1.0 + negative_exp)
    return probabilities


def attach_prediction_scores(frame, raw, teacher_id):
    """Prefer exported margins because float32 probabilities may be 0/1."""
    if len(frame) != len(raw):
        raise ValueError(f"Teacher {teacher_id} prediction columns have inconsistent lengths")

    probability_column = "prob" if "prob" in raw.columns else "base_prob"
    if probability_column in raw.columns:
        reported_probability = raw[probability_column].astype(float).to_numpy()
        if not np.isfinite(reported_probability).all():
            raise ValueError(f"Teacher {teacher_id} probabilities are non-finite")
        if ((reported_probability < 0.0) | (reported_probability > 1.0)).any():
            raise ValueError(f"Teacher {teacher_id} probabilities are outside [0, 1]")
        frame["oof_base_prob_reported"] = reported_probability
    else:
        reported_probability = None

    if "logit" in raw.columns:
        logits = raw["logit"].astype(float).to_numpy()
        if not np.isfinite(logits).all():
            raise ValueError(f"Teacher {teacher_id} logits are non-finite")
        frame["oof_base_logit"] = logits
        frame["oof_base_prob"] = stable_sigmoid(logits)
        frame["oof_score_source"] = "exported_logit"
        return frame

    if reported_probability is None:
        raise ValueError(f"Teacher {teacher_id} prediction CSV lacks logit/prob/base_prob")

    epsilon = 1e-6
    clipped = np.clip(reported_probability, epsilon, 1.0 - epsilon)
    frame["oof_base_prob"] = reported_probability
    frame["oof_base_logit"] = np.log(clipped / (1.0 - clipped))
    frame["oof_score_source"] = f"{probability_column}_fallback"
    return frame


def load_prediction(path, teacher_id):
    frame = normalize(pd.read_csv(path))
    raw = pd.read_csv(path)
    frame = attach_prediction_scores(frame, raw, teacher_id)
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Teacher {teacher_id} contains duplicate prediction keys")
    frame["oof_teacher_count"] = 1
    frame["oof_teacher_std"] = 0.0
    frame["oof_teacher_folds"] = str(teacher_id)
    return frame


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("protocol") != "outer-train-only-nested-dual-entity-blind-v1":
        raise ValueError("Unsupported or unsafe nested OOF manifest protocol")
    if manifest.get("outer_val_accessed") or manifest.get("outer_test_accessed"):
        raise ValueError("Manifest reports outer validation/test access")

    outer_train_path = Path(manifest["outer_train_path"])
    if sha256_file(outer_train_path) != manifest["outer_train_sha256"]:
        raise ValueError("Outer training CSV hash changed after split construction")
    source = normalize(pd.read_csv(outer_train_path, index_col=0))
    source = source.drop_duplicates(KEY_COLUMNS).reset_index(drop=True)
    feature_filter = manifest.get("embedding_filter", {})
    source = filter_missing_bindingdb_features(
        source,
        drug_ids=feature_filter.get("excluded_drug_ids", []),
        protein_ids=feature_filter.get("excluded_protein_ids", []),
    )

    specs = [parse_prediction_spec(spec) for spec in args.prediction]
    supplied_ids = [teacher_id for teacher_id, _ in specs]
    expected_ids = sorted(int(item["teacher_id"]) for item in manifest["teachers"])
    if sorted(supplied_ids) != expected_ids:
        raise ValueError(
            f"Expected predictions for teachers {expected_ids}, received {sorted(supplied_ids)}"
        )

    frames = []
    for teacher_id, prediction_path in specs:
        teacher = next(
            item for item in manifest["teachers"] if int(item["teacher_id"]) == teacher_id
        )
        dataset_dir = Path(teacher["dataset_dir"])
        teacher_train = normalize(
            pd.read_csv(dataset_dir / "blind_start" / "train_set0.csv", index_col=0)
        )
        teacher_val = normalize(
            pd.read_csv(dataset_dir / "blind_start" / "val_set0.csv", index_col=0)
        )
        target = normalize(
            pd.read_csv(dataset_dir / "blind_start" / "test_set0.csv", index_col=0)
        )
        prediction = load_prediction(prediction_path, teacher_id)

        expected = target.merge(prediction[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
        if not (expected["_merge"] == "both").all():
            raise ValueError(f"Teacher {teacher_id} predictions do not exactly cover its target block")

        seen_drugs = set(teacher_train["head"]) | set(teacher_val["head"])
        seen_proteins = set(teacher_train["tail"]) | set(teacher_val["tail"])
        if set(prediction["head"]) & seen_drugs:
            raise ValueError(f"Teacher {teacher_id} prediction drugs leak into teacher training")
        if set(prediction["tail"]) & seen_proteins:
            raise ValueError(f"Teacher {teacher_id} prediction proteins leak into teacher training")
        frames.append(prediction)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(KEY_COLUMNS).any():
        raise ValueError("A source row received multiple nested OOF predictions")
    membership = source.merge(combined[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    if not (membership["_merge"] == "both").all():
        raise ValueError("Nested OOF predictions do not cover the outer training split exactly once")

    combined["sample_weight"] = 1.0

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path)
    summary = {
        "protocol": manifest["protocol"],
        "outer_set_id": int(manifest["outer_set_id"]),
        "outer_train_sha256": manifest["outer_train_sha256"],
        "rows": int(len(combined)),
        "positive_rows": int(combined["label"].sum()),
        "negative_rows": int((combined["label"] == 0).sum()),
        "teacher_count": len(expected_ids),
        "outer_val_accessed": False,
        "outer_test_accessed": False,
        "output_csv": str(output_path.resolve()),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
