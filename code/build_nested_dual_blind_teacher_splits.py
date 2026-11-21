import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["head", "tail", "label"]
DEFAULT_MISSING_DRUG_IDS = {
    "141", "303", "836", "1167", "2837", "2957", "3663", "4612", "4723", "5534", "6258"
}
DEFAULT_MISSING_PROTEIN_IDS = {"2122", "901", "93"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build nested dual-entity-blind teacher splits from one outer training split."
    )
    parser.add_argument("--bindingdb-dir", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-buckets", type=int, default=3)
    parser.add_argument("--teacher-val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def normalize(frame):
    frame = frame.copy()
    frame["head"] = frame["head"].astype(str).str.replace("Drug:", "", regex=False)
    frame["tail"] = frame["tail"].astype(str).str.replace("Protein:", "", regex=False)
    frame["label"] = frame["label"].astype(int)
    return frame[KEY_COLUMNS]


def filter_missing_bindingdb_features(frame, drug_ids=None, protein_ids=None):
    drug_ids = DEFAULT_MISSING_DRUG_IDS if drug_ids is None else set(drug_ids)
    protein_ids = DEFAULT_MISSING_PROTEIN_IDS if protein_ids is None else set(protein_ids)
    keep = ~frame["head"].isin(drug_ids) & ~frame["tail"].isin(protein_ids)
    return frame.loc[keep].reset_index(drop=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_tiebreak(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def balanced_entity_buckets(values, num_buckets, seed):
    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2")
    counts = values.astype(str).value_counts()
    ordered = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), stable_tiebreak(item[0], seed)),
    )
    loads = [0] * num_buckets
    assignments = {}
    for entity, count in ordered:
        bucket = min(range(num_buckets), key=lambda index: (loads[index], index))
        assignments[str(entity)] = int(bucket)
        loads[bucket] += int(count)
    return assignments, loads


def stratified_train_val_split(frame, val_frac, seed):
    if not 0.0 < val_frac < 0.5:
        raise ValueError("teacher_val_frac must lie between 0 and 0.5")
    rng = np.random.RandomState(seed)
    val_indices = []
    for _, group in frame.groupby("label"):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * val_frac)))
        count = min(count, len(indices) - 1)
        if count <= 0:
            raise ValueError("Each class needs at least two teacher-training-pool rows")
        val_indices.extend(indices[:count].tolist())
    val_index = set(val_indices)
    train = frame.loc[[index for index in frame.index if index not in val_index]].copy()
    val = frame.loc[[index for index in frame.index if index in val_index]].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def build_teacher_frames(source, num_buckets=3, val_frac=0.1, seed=1234):
    source = normalize(source).drop_duplicates(KEY_COLUMNS).reset_index(drop=True)
    drug_buckets, drug_loads = balanced_entity_buckets(source["head"], num_buckets, seed)
    protein_buckets, protein_loads = balanced_entity_buckets(
        source["tail"], num_buckets, seed + 1
    )
    drug_bucket = source["head"].map(drug_buckets)
    protein_bucket = source["tail"].map(protein_buckets)

    teachers = []
    target_indices = []
    teacher_id = 0
    for heldout_drug_bucket in range(num_buckets):
        for heldout_protein_bucket in range(num_buckets):
            target_mask = (
                (drug_bucket == heldout_drug_bucket)
                & (protein_bucket == heldout_protein_bucket)
            )
            pool_mask = (
                (drug_bucket != heldout_drug_bucket)
                & (protein_bucket != heldout_protein_bucket)
            )
            target = source.loc[target_mask].copy().reset_index(drop=True)
            pool = source.loc[pool_mask].copy().reset_index(drop=True)
            if target.empty:
                raise ValueError(
                    "Empty target block for drug bucket "
                    f"{heldout_drug_bucket}, protein bucket {heldout_protein_bucket}"
                )
            train, val = stratified_train_val_split(
                pool,
                val_frac=val_frac,
                seed=seed + teacher_id,
            )

            train_drugs = set(train["head"]) | set(val["head"])
            train_proteins = set(train["tail"]) | set(val["tail"])
            if set(target["head"]) & train_drugs:
                raise AssertionError(f"Teacher {teacher_id} has held-out drug leakage")
            if set(target["tail"]) & train_proteins:
                raise AssertionError(f"Teacher {teacher_id} has held-out protein leakage")

            target_indices.extend(source.index[target_mask].tolist())
            teachers.append(
                {
                    "teacher_id": teacher_id,
                    "drug_bucket": heldout_drug_bucket,
                    "protein_bucket": heldout_protein_bucket,
                    "train": train,
                    "val": val,
                    "target": target,
                }
            )
            teacher_id += 1

    if sorted(target_indices) != list(range(len(source))):
        raise AssertionError("Nested teacher targets do not cover every outer-train row exactly once")
    return teachers, drug_buckets, protein_buckets, drug_loads, protein_loads


def write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def main():
    args = parse_args()
    bindingdb_dir = Path(args.bindingdb_dir)
    source_path = (
        bindingdb_dir / "blind_start" / f"train_set{args.outer_set_id}.csv"
    )
    source = normalize(pd.read_csv(source_path, index_col=0))
    raw_rows = len(source)
    source = source.drop_duplicates(KEY_COLUMNS).reset_index(drop=True)
    unique_rows_before_feature_filter = len(source)
    source = filter_missing_bindingdb_features(source)

    teachers, drug_buckets, protein_buckets, drug_loads, protein_loads = build_teacher_frames(
        source,
        num_buckets=args.num_buckets,
        val_frac=args.teacher_val_frac,
        seed=args.seed,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    teacher_summaries = []
    for teacher in teachers:
        teacher_id = teacher["teacher_id"]
        dataset_name = f"teacher_{teacher_id:02d}"
        dataset_dir = output_root / dataset_name
        split_dir = dataset_dir / "blind_start"
        write_frame(teacher["train"], split_dir / "train_set0.csv")
        write_frame(teacher["val"], split_dir / "val_set0.csv")
        # This inner test is a subset of the outer training split, not the outer test split.
        write_frame(teacher["target"], split_dir / "test_set0.csv")

        embedding_link = dataset_dir / "embeddings"
        if not embedding_link.exists() and not embedding_link.is_symlink():
            embedding_link.symlink_to(
                (bindingdb_dir / "embeddings").resolve(),
                target_is_directory=True,
            )

        teacher_summaries.append(
            {
                "teacher_id": teacher_id,
                "dataset_name": dataset_name,
                "dataset_dir": str(dataset_dir.resolve()),
                "drug_bucket": teacher["drug_bucket"],
                "protein_bucket": teacher["protein_bucket"],
                "train_rows": int(len(teacher["train"])),
                "val_rows": int(len(teacher["val"])),
                "target_rows": int(len(teacher["target"])),
                "target_positive_rows": int(teacher["target"]["label"].sum()),
                "target_train_drug_overlap": 0,
                "target_train_protein_overlap": 0,
            }
        )

    pd.DataFrame(
        [{"entity": entity, "bucket": bucket} for entity, bucket in drug_buckets.items()]
    ).to_csv(output_root / "drug_bucket_assignments.csv", index=False)
    pd.DataFrame(
        [{"entity": entity, "bucket": bucket} for entity, bucket in protein_buckets.items()]
    ).to_csv(output_root / "protein_bucket_assignments.csv", index=False)

    manifest = {
        "protocol": "outer-train-only-nested-dual-entity-blind-v1",
        "outer_set_id": int(args.outer_set_id),
        "outer_train_path": str(source_path.resolve()),
        "outer_train_sha256": sha256_file(source_path),
        "outer_raw_rows": int(raw_rows),
        "outer_unique_rows": int(len(source)),
        "outer_unique_rows_before_feature_filter": int(unique_rows_before_feature_filter),
        "embedding_filter": {
            "excluded_drug_ids": sorted(DEFAULT_MISSING_DRUG_IDS),
            "excluded_protein_ids": sorted(DEFAULT_MISSING_PROTEIN_IDS),
        },
        "outer_val_accessed": False,
        "outer_test_accessed": False,
        "num_buckets": int(args.num_buckets),
        "teacher_count": int(len(teachers)),
        "teacher_val_frac": float(args.teacher_val_frac),
        "seed": int(args.seed),
        "drug_bucket_row_loads": drug_loads,
        "protein_bucket_row_loads": protein_loads,
        "target_coverage_rows": int(sum(item["target_rows"] for item in teacher_summaries)),
        "teachers": teacher_summaries,
    }
    manifest_path = output_root / "nested_oof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
