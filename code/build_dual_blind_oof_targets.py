import argparse
import json
import math
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["head", "tail", "label"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build dual-entity blind OOF targets for an outer BindingDB fold "
            "from completed blind-start teacher test predictions."
        )
    )
    parser.add_argument("--bindingdb-dir", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="FOLD=CSV",
        help="Teacher fold id and its exported blind-start test predictions.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--minimum-coverage", type=float, default=0.45)
    return parser.parse_args()


def normalize(frame):
    frame = frame.copy()
    frame["head"] = frame["head"].astype(str).str.replace("Drug:", "", regex=False)
    frame["tail"] = frame["tail"].astype(str).str.replace("Protein:", "", regex=False)
    frame["label"] = frame["label"].astype(int)
    return frame


def parse_prediction_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Prediction spec must be FOLD=CSV, received: {spec}")
    fold_text, path_text = spec.split("=", 1)
    return int(fold_text), Path(path_text)


def assert_dual_entity_blind(predictions, teacher_train, fold):
    train_drugs = set(teacher_train["head"])
    train_proteins = set(teacher_train["tail"])
    leaked = predictions["head"].isin(train_drugs) | predictions["tail"].isin(train_proteins)
    if leaked.any():
        examples = predictions.loc[leaked, KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(
            f"Teacher fold {fold} is not dual-entity blind for {int(leaked.sum())} rows: {examples}"
        )


def load_teacher_predictions(bindingdb_dir, fold, prediction_path):
    if not prediction_path.exists():
        raise FileNotFoundError(f"Teacher prediction file not found: {prediction_path}")

    predictions = normalize(pd.read_csv(prediction_path))
    probability_column = "prob" if "prob" in predictions.columns else "base_prob"
    if probability_column not in predictions.columns:
        raise ValueError(f"Teacher fold {fold} predictions lack prob/base_prob")
    predictions = predictions[KEY_COLUMNS + [probability_column]].rename(
        columns={probability_column: "teacher_probability"}
    )
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Teacher fold {fold} has duplicate prediction keys")
    if not predictions["teacher_probability"].between(0.0, 1.0).all():
        raise ValueError(f"Teacher fold {fold} probabilities are outside [0, 1]")

    teacher_train = normalize(
        pd.read_csv(bindingdb_dir / "blind_start" / f"train_set{fold}.csv", index_col=0)
    )
    assert_dual_entity_blind(predictions, teacher_train, fold)

    teacher_test = normalize(
        pd.read_csv(bindingdb_dir / "blind_start" / f"test_set{fold}.csv", index_col=0)
    )
    test_keys = teacher_test[KEY_COLUMNS].drop_duplicates()
    membership = predictions.merge(test_keys, on=KEY_COLUMNS, how="left", indicator=True)
    outside_test = membership["_merge"] != "both"
    if outside_test.any():
        raise ValueError(
            f"Teacher fold {fold} predictions contain {int(outside_test.sum())} rows outside its test split"
        )

    predictions["teacher_fold"] = int(fold)
    return predictions


def aggregate_teachers(predictions):
    combined = pd.concat(predictions, ignore_index=True)

    def fold_ids(values):
        return ",".join(str(value) for value in sorted(set(int(item) for item in values)))

    aggregate = combined.groupby(KEY_COLUMNS, as_index=False).agg(
        oof_base_prob=("teacher_probability", "mean"),
        oof_teacher_std=("teacher_probability", lambda values: float(values.std(ddof=0))),
        oof_teacher_count=("teacher_fold", "nunique"),
        oof_teacher_folds=("teacher_fold", fold_ids),
    )
    epsilon = 1e-6
    clipped = aggregate["oof_base_prob"].clip(epsilon, 1.0 - epsilon)
    aggregate["oof_base_logit"] = clipped.map(
        lambda probability: math.log(probability / (1.0 - probability))
    )
    return aggregate


def main():
    args = parse_args()
    bindingdb_dir = Path(args.bindingdb_dir)
    specs = [parse_prediction_spec(spec) for spec in args.prediction]
    folds = [fold for fold, _ in specs]
    if len(folds) != len(set(folds)):
        raise ValueError("Each teacher fold may be supplied only once")
    if args.outer_set_id in folds:
        raise ValueError("The outer fold teacher is in-sample and cannot provide router OOF targets")

    source_path = bindingdb_dir / "blind_start" / f"train_set{args.outer_set_id}.csv"
    source = normalize(pd.read_csv(source_path, index_col=0))
    raw_rows = len(source)
    source = source.drop_duplicates(KEY_COLUMNS).reset_index(drop=True)

    teacher_frames = [
        load_teacher_predictions(bindingdb_dir, fold, prediction_path)
        for fold, prediction_path in specs
    ]
    aggregate = aggregate_teachers(teacher_frames)
    merged = source.merge(aggregate, on=KEY_COLUMNS, how="left", validate="one_to_one")
    covered = merged["oof_base_prob"].notna()
    coverage = float(covered.mean())
    if coverage < args.minimum_coverage:
        raise ValueError(
            f"Dual-blind OOF coverage {coverage:.4f} is below required {args.minimum_coverage:.4f}"
        )
    output = merged.loc[covered].reset_index(drop=True)
    output["sample_weight"] = 1.0

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path)

    summary = {
        "outer_set_id": args.outer_set_id,
        "source_csv": str(source_path.resolve()),
        "source_raw_rows": int(raw_rows),
        "source_unique_rows": int(len(source)),
        "covered_rows": int(len(output)),
        "coverage": coverage,
        "positive_rows": int(output["label"].sum()),
        "negative_rows": int((1 - output["label"]).sum()),
        "teacher_folds": sorted(folds),
        "teacher_count_distribution": {
            str(int(count)): int(rows)
            for count, rows in output["oof_teacher_count"].value_counts().sort_index().items()
        },
        "mean_teacher_std": float(output["oof_teacher_std"].mean()),
        "output_csv": str(output_path.resolve()),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
