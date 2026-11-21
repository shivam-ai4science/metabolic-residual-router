import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble a BindingDB dataset with OOF router targets.")
    parser.add_argument("--bindingdb-dir", required=True)
    parser.add_argument("--oof-train-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set-id", type=int, required=True)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Copy the outer test split only for a locked final-evaluation dataset.",
    )
    return parser.parse_args()


def parse_teacher_folds(value):
    folds = [int(part) for part in str(value).split(",") if part.strip()]
    if not folds:
        raise ValueError(f"Invalid oof_teacher_folds value: {value!r}")
    return sorted(set(folds))


def validate_oof_train(train):
    required = {
        "head",
        "tail",
        "label",
        "oof_base_prob",
        "oof_base_logit",
        "oof_teacher_count",
        "oof_teacher_std",
        "oof_teacher_folds",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"OOF training CSV is missing columns: {missing}")

    if train.empty:
        raise ValueError("OOF training CSV is empty")
    if not train["oof_base_prob"].between(0.0, 1.0).all():
        raise ValueError("oof_base_prob must be in [0, 1]")
    if (train["oof_teacher_count"] < 1).any():
        raise ValueError("oof_teacher_count must be at least 1")
    if (train["oof_teacher_std"] < 0).any():
        raise ValueError("oof_teacher_std must be non-negative")

    teacher_sets = train["oof_teacher_folds"].map(parse_teacher_folds)
    observed_counts = teacher_sets.map(len)
    expected_counts = train["oof_teacher_count"].astype(int)
    if not observed_counts.equals(expected_counts):
        mismatch = train.loc[observed_counts != expected_counts].iloc[0]
        raise ValueError(
            "oof_teacher_count does not match oof_teacher_folds for "
            f"head={mismatch['head']} tail={mismatch['tail']}"
        )
    return teacher_sets


def main():
    args = parse_args()
    source = Path(args.bindingdb_dir)
    output = Path(args.output_dir)
    split_dir = output / "blind_start"
    split_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.oof_train_csv, index_col=0)
    teacher_sets = validate_oof_train(train)
    train.to_csv(split_dir / f"train_set{args.set_id}.csv")

    splits = ["val"] + (["test"] if args.include_test else [])
    for split in splits:
        source_path = source / "blind_start" / f"{split}_set{args.set_id}.csv"
        shutil.copy2(source_path, split_dir / source_path.name)

    embedding_link = output / "embeddings"
    if not embedding_link.exists() and not embedding_link.is_symlink():
        embedding_link.symlink_to((source / "embeddings").resolve(), target_is_directory=True)

    teacher_folds = sorted({fold for folds in teacher_sets for fold in folds})
    summary = {
        "rows": int(len(train)),
        "positive_rows": int(train["label"].sum()),
        "negative_rows": int((train["label"] == 0).sum()),
        "teacher_folds": teacher_folds,
        "teacher_fold_count": len(teacher_folds),
        "mean_teachers_per_row": float(train["oof_teacher_count"].mean()),
        "mean_teacher_std": float(train["oof_teacher_std"].mean()),
        "test_split_included": bool(args.include_test),
        "embedding_target": str(embedding_link.resolve()),
        "output_dir": str(output.resolve()),
    }
    (output / "router_oof_dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
