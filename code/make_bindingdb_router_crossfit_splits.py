import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create entity-disjoint inner folds for out-of-fold router teacher predictions."
    )
    parser.add_argument("--bindingdb-dir", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--group-by", choices=["protein", "drug"], default="protein")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--copy-embeddings", action="store_true")
    return parser.parse_args()


def assign_groups(frame, group_column, num_folds, seed):
    grouped = frame.groupby(group_column)["label"].agg(["size", "sum"]).reset_index()
    grouped = grouped.sample(frac=1.0, random_state=seed).sort_values("size", ascending=False)
    target_rows = len(frame) / num_folds
    target_pos = max(float(frame["label"].sum()) / num_folds, 1.0)
    fold_rows = np.zeros(num_folds, dtype=float)
    fold_pos = np.zeros(num_folds, dtype=float)
    assignments = {}

    for _, row in grouped.iterrows():
        best_fold = None
        best_objective = None
        for fold in range(num_folds):
            candidate_rows = fold_rows.copy()
            candidate_pos = fold_pos.copy()
            candidate_rows[fold] += float(row["size"])
            candidate_pos[fold] += float(row["sum"])
            objective = (
                np.square((candidate_rows - target_rows) / max(target_rows, 1.0)).sum()
                + np.square((candidate_pos - target_pos) / target_pos).sum()
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_fold = fold
        assignments[str(row[group_column])] = int(best_fold)
        fold_rows[best_fold] += float(row["size"])
        fold_pos[best_fold] += float(row["sum"])
    return assignments


def copy_or_link_embeddings(source_dir, destination_dir, copy_embeddings):
    source = source_dir / "embeddings"
    destination = destination_dir / "embeddings"
    if destination.exists() or destination.is_symlink():
        return
    if copy_embeddings:
        shutil.copytree(source, destination)
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError as error:
        raise OSError(
            f"Could not create embedding symlink {destination}. "
            "Run on Linux/A100 or pass --copy-embeddings if disk space permits."
        ) from error


def main():
    args = parse_args()
    source_dir = Path(args.bindingdb_dir)
    source_csv = source_dir / "blind_start" / f"train_set{args.outer_set_id}.csv"
    frame = pd.read_csv(source_csv, index_col=0).drop_duplicates().reset_index(drop=True)
    group_column = "tail" if args.group_by == "protein" else "head"
    assignments = assign_groups(frame, group_column, args.num_folds, args.seed)
    row_fold = frame[group_column].astype(str).map(assignments)
    if row_fold.isna().any():
        raise RuntimeError("Some rows were not assigned to an inner fold")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold in range(args.num_folds):
        dataset_dir = output_root / f"BindingDB_AIBind_outer{args.outer_set_id}_{args.group_by}_crossfit{fold}"
        split_dir = dataset_dir / "blind_start"
        split_dir.mkdir(parents=True, exist_ok=True)
        holdout = frame[row_fold == fold].reset_index(drop=True)
        teacher_train = frame[row_fold != fold].reset_index(drop=True)
        teacher_train.to_csv(split_dir / "train_set0.csv")
        holdout.to_csv(split_dir / "val_set0.csv")
        holdout.to_csv(split_dir / "test_set0.csv")
        copy_or_link_embeddings(source_dir, dataset_dir, args.copy_embeddings)

        train_groups = set(teacher_train[group_column].astype(str))
        holdout_groups = set(holdout[group_column].astype(str))
        overlap = len(train_groups & holdout_groups)
        if overlap:
            raise RuntimeError(f"Inner fold {fold} has {overlap} leaked {args.group_by} groups")
        summaries.append(
            {
                "fold": fold,
                "dataset_dir": str(dataset_dir.resolve()),
                "train_rows": int(len(teacher_train)),
                "holdout_rows": int(len(holdout)),
                "train_pos": int(teacher_train["label"].sum()),
                "holdout_pos": int(holdout["label"].sum()),
                "group_overlap": overlap,
            }
        )

    manifest = {
        "source_csv": str(source_csv.resolve()),
        "outer_set_id": args.outer_set_id,
        "num_folds": args.num_folds,
        "group_by": args.group_by,
        "seed": args.seed,
        "folds": summaries,
    }
    manifest_path = output_root / f"outer{args.outer_set_id}_{args.group_by}_crossfit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
