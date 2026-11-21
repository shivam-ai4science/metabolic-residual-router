#!/usr/bin/env python3
"""Verify the compact five-fold V3.7 run receipts and locked metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts/v37_fivefold_compact"
RESULTS = ROOT / "results/results_five_fold.csv"
SPLIT_AUDIT = ROOT / "results/final_5fold_local_audit.json"
OUTPUT = ROOT / "results/fivefold_receipt_audit.json"

EXPECTED_TEST_ROWS = {0: 7869, 1: 8366, 2: 7902, 3: 8125, 4: 7930}
FINAL_EXPORTS = {
    0: ("frozen_outer_test_epoch37_deduplicated/test_predictions.csv", "logit"),
    1: ("frozen_outer_test_epoch37/test_routed_analysis.csv", "final_logit"),
    2: ("frozen_outer_test_epoch26/test_routed_analysis.csv", "final_logit"),
    3: ("frozen_outer_test_epoch38/test_routed_analysis.csv", "final_logit"),
    4: ("frozen_test_epoch37/test_routed_analysis.csv", "final_logit"),
}
METRIC_TOLERANCE = 7e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def normalize_entity(value: str) -> str:
    normalized = str(value).strip().split(":")[-1]
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def row_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        normalize_entity(row["head"]),
        normalize_entity(row["tail"]),
        int(float(row["label"])),
    )


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    assert positives and negatives
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    assert positives
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    previous_precision = 1.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        true_positives += sum(label for _, label in group)
        false_positives += len(group) - sum(label for _, label in group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * (precision + previous_precision) / 2.0
        previous_recall = recall
        previous_precision = precision
        index = end
    return area


def verify_internal_checksums() -> int:
    manifest = RECEIPTS / "SHA256SUMS.txt"
    assert manifest.is_file(), manifest
    entries: dict[str, str] = {}
    for raw_line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = raw_line.split(maxsplit=1)
        relative = relative.lstrip("*")
        posix_path = PurePosixPath(relative)
        assert not posix_path.is_absolute() and ".." not in posix_path.parts, relative
        assert relative not in entries, relative
        path = RECEIPTS.joinpath(*posix_path.parts)
        assert path.is_file(), path
        actual = sha256(path)
        assert actual == expected, (relative, actual, expected)
        entries[relative] = expected

    actual_files = {
        path.relative_to(RECEIPTS).as_posix()
        for path in RECEIPTS.rglob("*")
        if path.is_file() and path != manifest
    }
    assert actual_files == set(entries), (
        sorted(actual_files - set(entries)),
        sorted(set(entries) - actual_files),
    )
    return len(entries)


def accepted_results() -> dict[int, dict[str, str]]:
    rows = read_csv(RESULTS)
    fold_rows = {
        int(row["fold"]): row
        for row in rows
        if row["fold"].strip().lower() != "mean"
    }
    assert sorted(fold_rows) == [0, 1, 2, 3, 4]
    return fold_rows


def selected_epoch(metrics_path: Path, metric: str) -> tuple[int, float]:
    rows = read_csv(metrics_path)
    candidates = [
        (int(float(row["epoch"])), float(row[metric]))
        for row in rows
        if row.get("epoch", "").strip() and row.get(metric, "").strip()
    ]
    assert candidates, (metrics_path, metric)
    best_value = max(value for _, value in candidates)
    best_epochs = {epoch for epoch, value in candidates if value == best_value}
    assert len(best_epochs) == 1, (metrics_path, metric, best_epochs)
    return next(iter(best_epochs)), best_value


def checkpoint_epoch(name: str) -> int:
    match = re.fullmatch(r"best_model_epoch(\d+)\.ckpt", name)
    assert match, name
    return int(match.group(1))


def assert_no_training_test_metrics(metrics_path: Path) -> None:
    rows = read_csv(metrics_path)
    if not rows:
        return
    test_columns = [name for name in rows[0] if name.startswith("test_")]
    assert all(not row.get(name, "").strip() for row in rows for name in test_columns), metrics_path


def audit_split_receipt(inventory: dict[str, object]) -> int:
    identities = inventory["benchmark_split_identities"]
    assert isinstance(identities, list) and len(identities) == 15
    identity_map = {
        (int(item["fold"]), str(item["split"])): str(item["sha256"])
        for item in identities
    }
    assert set(identity_map) == {
        (fold, split)
        for fold in range(5)
        for split in ("train", "val", "test")
    }

    locked = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))
    locked_by_name = {
        Path(path).name: digest
        for path, digest in locked["split_hashes"].items()
    }
    for fold in range(5):
        for split in ("train", "val", "test"):
            assert locked_by_name[f"{split}_set{fold}.csv"] == identity_map[(fold, split)]
        fold_audit = next(item for item in locked["fold_audits"] if int(item["fold"]) == fold)
        assert fold_audit["train_test_pair_overlap"] == 0
        assert fold_audit["train_test_compound_overlap"] == 0
        assert fold_audit["train_test_protein_overlap"] == 0
        assert fold_audit["test_rows_after_feature_exclusions"] == EXPECTED_TEST_ROWS[fold]
    return len(identity_map)


def main() -> None:
    checksum_entries = verify_internal_checksums()
    inventory = json.loads((RECEIPTS / "INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["protocol"] == "v37-fivefold-compact-run-receipts-v1"
    assert inventory["folds"] == [0, 1, 2, 3, 4]
    assert inventory["checkpoint_weights_included"] is False
    assert inventory["datasets_included"] is False
    assert inventory["embeddings_included"] is False
    assert not list(RECEIPTS.rglob("*.ckpt"))
    assert not list(RECEIPTS.rglob("*.pt"))
    assert not list(RECEIPTS.rglob("*.pth"))

    checkpoint_identities = inventory["checkpoint_identities"]
    assert isinstance(checkpoint_identities, list) and len(checkpoint_identities) == 55
    stage_counts = Counter(str(item["stage"]) for item in checkpoint_identities)
    assert stage_counts == Counter({"teacher": 45, "stage1": 5, "stage2": 5})
    split_identity_count = audit_split_receipt(inventory)
    results = accepted_results()

    fold_audits = []
    for fold in range(5):
        nested_path = RECEIPTS / f"manifests/fold{fold}/nested_oof_manifest.json"
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        assert nested["outer_set_id"] == fold
        assert nested["teacher_count"] == 9
        assert nested["outer_val_accessed"] is False
        assert nested["outer_test_accessed"] is False
        assert nested["target_coverage_rows"] == nested["outer_unique_rows"]
        assert len(nested["teachers"]) == 9

        all_teacher_keys: Counter[tuple[str, str, int]] = Counter()
        teacher_checkpoint_epochs = []
        for teacher in nested["teachers"]:
            teacher_id = int(teacher["teacher_id"])
            tag = f"{teacher_id:02d}"
            teacher_dir = RECEIPTS / f"teachers/fold{fold}/teacher_{tag}"
            required = {
                "launch_config.yaml",
                "run_config.yaml",
                "metrics.csv",
                "test_predictions.csv",
            }
            assert required == {path.name for path in teacher_dir.iterdir() if path.is_file()}
            assert teacher["target_train_drug_overlap"] == 0
            assert teacher["target_train_protein_overlap"] == 0
            config_text = (teacher_dir / "run_config.yaml").read_text(encoding="utf-8")
            assert f"BindingDB_AIBind_outer{fold}_nested_oof/teacher_{tag}" in config_text

            prediction_rows = read_csv(teacher_dir / "test_predictions.csv")
            assert len(prediction_rows) == int(teacher["target_rows"])
            assert sum(int(float(row["label"])) for row in prediction_rows) == int(
                teacher["target_positive_rows"]
            )
            all_teacher_keys.update(row_key(row) for row in prediction_rows)

            identity = next(
                item
                for item in checkpoint_identities
                if item["fold"] == fold
                and item["stage"] == "teacher"
                and item["teacher"] == teacher_id
            )
            expected_epoch = checkpoint_epoch(str(identity["checkpoint"]))
            best_epoch, _ = selected_epoch(teacher_dir / "metrics.csv", "val_score")
            assert best_epoch == expected_epoch, (fold, teacher_id, best_epoch, expected_epoch)
            teacher_checkpoint_epochs.append(expected_epoch)

        assert sum(all_teacher_keys.values()) == nested["target_coverage_rows"]
        assert len(all_teacher_keys) == nested["target_coverage_rows"]

        report_dir = RECEIPTS / f"reports/v37_nested_oof_outer{fold}"
        raw_oof = read_csv(
            report_dir / f"BindingDB_AIBind_outer{fold}_nested_dual_blind_oof_train.csv"
        )
        calibrated_oof = read_csv(
            report_dir / f"BindingDB_AIBind_outer{fold}_nested_dual_blind_oof_train_calibrated.csv"
        )
        assert Counter(row_key(row) for row in raw_oof) == all_teacher_keys
        assert Counter(row_key(row) for row in calibrated_oof) == all_teacher_keys
        oof_summary = json.loads(
            (report_dir / f"BindingDB_AIBind_outer{fold}_nested_dual_blind_oof_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert oof_summary["outer_set_id"] == fold
        assert oof_summary["rows"] == nested["target_coverage_rows"]
        assert oof_summary["outer_val_accessed"] is False
        assert oof_summary["outer_test_accessed"] is False

        router_summary = json.loads(
            (RECEIPTS / f"manifests/fold{fold}/router_oof_dataset_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert router_summary["rows"] == nested["target_coverage_rows"]
        assert router_summary["teacher_fold_count"] == 9
        assert router_summary["mean_teachers_per_row"] == 1.0

        stage1_identity = next(
            item
            for item in checkpoint_identities
            if item["fold"] == fold and item["stage"] == "stage1"
        )
        stage1_dir = RECEIPTS / "runs" / str(stage1_identity["run"])
        stage1_epoch, _ = selected_epoch(stage1_dir / "version_0/metrics.csv", "val_metabolic_score")
        assert stage1_epoch == checkpoint_epoch(str(stage1_identity["checkpoint"]))

        stage2_identity = next(
            item
            for item in checkpoint_identities
            if item["fold"] == fold and item["stage"] == "stage2"
        )
        stage2_dir = RECEIPTS / "runs" / str(stage2_identity["run"])
        stage2_config = (stage2_dir / "run_config.yaml").read_text(encoding="utf-8")
        assert re.search(r"^\s*test_after_fit:\s*false\s*$", stage2_config, flags=re.MULTILINE)
        stage2_metrics = stage2_dir / "version_0/metrics.csv"
        stage2_epoch, stage2_score = selected_epoch(stage2_metrics, "val_score")
        assert stage2_epoch == checkpoint_epoch(str(stage2_identity["checkpoint"]))
        assert_no_training_test_metrics(stage2_metrics)

        export_relative, score_column = FINAL_EXPORTS[fold]
        export_path = stage2_dir / export_relative
        prediction_rows = read_csv(export_path)
        assert len(prediction_rows) == EXPECTED_TEST_ROWS[fold]
        prediction_keys = Counter(row_key(row) for row in prediction_rows)
        assert len(prediction_keys) == len(prediction_rows)
        labels = [int(float(row["label"])) for row in prediction_rows]
        scores = [float(row[score_column]) for row in prediction_rows]
        assert all(math.isfinite(score) for score in scores)
        measured_auc = roc_auc(labels, scores)
        measured_aupr = pr_auc(labels, scores)
        expected_auc = float(results[fold]["v37_auc"])
        expected_aupr = float(results[fold]["v37_aupr"])
        assert abs(measured_auc - expected_auc) <= METRIC_TOLERANCE, (
            fold,
            measured_auc,
            expected_auc,
        )
        assert abs(measured_aupr - expected_aupr) <= METRIC_TOLERANCE, (
            fold,
            measured_aupr,
            expected_aupr,
        )

        fold_audits.append(
            {
                "fold": fold,
                "outer_train_target_rows": nested["target_coverage_rows"],
                "teacher_count": 9,
                "teacher_prediction_rows": sum(all_teacher_keys.values()),
                "teacher_checkpoint_epochs": teacher_checkpoint_epochs,
                "stage1_run": stage1_identity["run"],
                "stage1_checkpoint": stage1_identity["checkpoint"],
                "stage2_run": stage2_identity["run"],
                "stage2_checkpoint": stage2_identity["checkpoint"],
                "stage2_selected_epoch": stage2_epoch,
                "stage2_validation_score": stage2_score,
                "frozen_prediction_export": export_relative,
                "test_rows": len(prediction_rows),
                "positive_rows": sum(labels),
                "v37_auc": measured_auc,
                "v37_aupr": measured_aupr,
                "outer_val_accessed_by_nested_targets": False,
                "outer_test_accessed_by_nested_targets": False,
                "test_metrics_present_during_stage2_fit": False,
            }
        )

    payload = {
        "protocol": "public-v37-fivefold-compact-receipt-audit-v1",
        "receipt_protocol": inventory["protocol"],
        "internal_checksum_entries_verified": checksum_entries,
        "benchmark_split_identities_verified": split_identity_count,
        "checkpoint_identities_verified": len(checkpoint_identities),
        "checkpoint_weights_included": False,
        "folds": fold_audits,
        "verdict": "FIVE_FOLD_COMPACT_RECEIPT_AUDIT_OK",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print("FIVE-FOLD COMPACT RECEIPT AUDIT OK")
    for item in fold_audits:
        print(
            f"fold={item['fold']} rows={item['test_rows']} "
            f"AUC={item['v37_auc']:.6f} AUPR={item['v37_aupr']:.6f} "
            f"checkpoint={item['stage2_checkpoint']}"
        )


if __name__ == "__main__":
    main()
