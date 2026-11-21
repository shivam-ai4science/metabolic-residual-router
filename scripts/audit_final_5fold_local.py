#!/usr/bin/env python3
"""Audit supplied benchmark splits against the locked five-fold identities."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "DPI/datasets/BindingDB_AIBind/blind_start"
RESULTS = ROOT / "results/results_five_fold.csv"
OUTPUT = ROOT / "results/final_5fold_local_audit.json"
EVALUATED_COHORT_ROWS = {0: 7869, 1: 8366, 2: 7902, 3: 8125, 4: 7930}
DRUG_EXCLUDE = {"141", "303", "836", "1167", "2837", "2957", "3663", "4612", "4723", "5534", "6258"}
PROTEIN_EXCLUDE = {"2122", "901", "93"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[tuple[str, str, int]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return [(row["head"], row["tail"], int(float(row["label"]))) for row in rows]


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        results = [
            row
            for row in csv.DictReader(stream)
            if row["fold"].strip().lower() != "mean"
        ]
    result_folds = [int(row["fold"]) for row in results]
    assert sorted(result_folds) == [0, 1, 2, 3, 4]
    assert len(result_folds) == len(set(result_folds)) == 5
    assert all(row["status"] == "complete" for row in results)

    fold_audits = []
    split_hashes = {}
    for fold in range(5):
        train_path = SPLITS / f"train_set{fold}.csv"
        val_path = SPLITS / f"val_set{fold}.csv"
        test_path = SPLITS / f"test_set{fold}.csv"
        for path in (train_path, val_path, test_path):
            assert path.is_file(), path
            split_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        train = read_split(train_path)
        val = read_split(val_path)
        test = read_split(test_path)
        train_pairs = Counter((head, tail, label) for head, tail, label in train)
        test_pairs = Counter((head, tail, label) for head, tail, label in test)
        deduplicated_test_rows = len(test_pairs)
        evaluated_test = [
            row
            for row in test_pairs
            if row[0].replace("Drug:", "") not in DRUG_EXCLUDE
            and row[1].replace("Protein:", "") not in PROTEIN_EXCLUDE
        ]
        train_heads = {row[0] for row in train}
        train_tails = {row[1] for row in train}
        test_heads = {row[0] for row in test}
        test_tails = {row[1] for row in test}
        pair_overlap = sum((train_pairs & test_pairs).values())
        head_overlap = len(train_heads & test_heads)
        tail_overlap = len(train_tails & test_tails)
        assert pair_overlap == head_overlap == tail_overlap == 0, (fold, pair_overlap, head_overlap, tail_overlap)
        assert len(evaluated_test) == EVALUATED_COHORT_ROWS[fold], (
            fold,
            len(evaluated_test),
            EVALUATED_COHORT_ROWS[fold],
        )
        fold_audits.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows_raw_csv": len(test),
                "test_positive_rows_raw_csv": sum(row[2] for row in test),
                "test_rows_after_exact_triple_deduplication": deduplicated_test_rows,
                "duplicates_removed_by_loader_semantics": len(test) - deduplicated_test_rows,
                "test_rows_after_feature_exclusions": len(evaluated_test),
                "feature_excluded_rows_after_deduplication": deduplicated_test_rows - len(evaluated_test),
                "train_test_pair_overlap": pair_overlap,
                "train_test_compound_overlap": head_overlap,
                "train_test_protein_overlap": tail_overlap,
            }
        )

    payload = {
        "protocol": "local-five-fold-definition-and-accepted-table-audit-v1",
        "results_source": str(RESULTS.relative_to(ROOT)).replace("\\", "/"),
        "results_sha256": sha256(RESULTS),
        "result_folds_exact_once": result_folds,
        "all_status_complete": True,
        "split_hashes": split_hashes,
        "fold_audits": fold_audits,
        "run_level_scope": (
            "This audit verifies supplied benchmark split definitions. Five-fold compact run receipts, "
            "frozen predictions, and training/model-selection evidence are verified separately by "
            "scripts/audit_fivefold_receipts.py."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print("LOCAL FIVE-FOLD SPLIT/RESULT AUDIT OK")


if __name__ == "__main__":
    main()
