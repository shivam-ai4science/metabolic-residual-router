import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import yaml

from build_fold_safe_metabolic_dataset import (
    load_bindingdb_features,
    load_bindingdb_fold,
    load_metabolic_dataset,
)
from build_nested_dual_blind_teacher_splits import (
    KEY_COLUMNS,
    filter_missing_bindingdb_features,
    normalize,
    sha256_file,
)
from src.losses import get_loss
from src.models import get_model
from train import audit_checkpoint_compatibility, load_dataframes
from utils import coerce_types


def parse_args():
    parser = argparse.ArgumentParser(description="Fail-closed V3.7 Stage-2 leakage audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bindingdb-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--calibrated-oof", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--metabolic-dataset-dir", required=True)
    parser.add_argument("--outer-set-id", type=int, required=True)
    parser.add_argument(
        "--allow-present-unloaded-test",
        action="store_true",
        help=(
            "Retrospective-only: permit an exact copy of the outer test split to "
            "exist in the router dataset after proving that the historical run "
            "did not request or record test evaluation during fitting."
        ),
    )
    return parser.parse_args()


def key_set(frame):
    return set(map(tuple, normalize(frame)[KEY_COLUMNS].itertuples(index=False, name=None)))


def load_csv(path):
    return pd.read_csv(path, index_col=0)


def main():
    args = parse_args()
    fold = int(args.outer_set_id)
    config = coerce_types(yaml.safe_load(Path(args.config).read_text()))
    manifest = json.loads(Path(args.manifest).read_text(encoding="ascii"))
    calibration = json.loads(Path(args.calibration_summary).read_text(encoding="ascii"))
    bindingdb = Path(args.bindingdb_dir)
    dataset = Path(config["data"]["base_path"]) / config["data"]["dataset_name"]

    assert manifest["outer_set_id"] == fold
    assert not manifest["outer_val_accessed"] and not manifest["outer_test_accessed"]
    assert not calibration["outer_val_accessed"] and not calibration["outer_test_accessed"]
    assert config["data"]["set_id"] == fold
    assert f"outer{fold}" in config["data"]["dataset_name"]
    assert f"fold{fold}" in config["finetune"]["source_run"]
    assert config["evaluation"]["test_after_fit"] is False
    assert config["loss"]["params"]["require_oof_targets"] is True
    assert calibration["output_sha256"] == sha256_file(args.calibrated_oof)
    assert config["calibration"]["output_sha256"] == calibration["output_sha256"]

    outer_train_path = Path(manifest["outer_train_path"])
    assert sha256_file(outer_train_path) == manifest["outer_train_sha256"]
    outer_train = normalize(load_csv(outer_train_path)).drop_duplicates(KEY_COLUMNS)
    excluded = manifest.get("embedding_filter", {})
    outer_train = filter_missing_bindingdb_features(
        outer_train, excluded.get("excluded_drug_ids", []), excluded.get("excluded_protein_ids", [])
    )
    router_train = load_csv(dataset / "blind_start" / f"train_set{fold}.csv")
    calibrated = load_csv(args.calibrated_oof)
    assert key_set(router_train) == key_set(calibrated) == key_set(outer_train)
    assert len(router_train) == len(calibrated) == len(outer_train)

    source_val = bindingdb / "blind_start" / f"val_set{fold}.csv"
    router_val = dataset / "blind_start" / f"val_set{fold}.csv"
    assert sha256_file(source_val) == sha256_file(router_val)
    source_test_path = bindingdb / "blind_start" / f"test_set{fold}.csv"
    router_test_path = dataset / "blind_start" / f"test_set{fold}.csv"
    if router_test_path.exists():
        assert args.allow_present_unloaded_test
        assert Path(args.config).name == "run_config.yaml"
        assert config["evaluation"]["test_after_fit"] is False
        assert sha256_file(source_test_path) == sha256_file(router_test_path)

        router_summary_path = dataset / "router_oof_dataset_summary.json"
        assert router_summary_path.is_file()
        router_summary = json.loads(router_summary_path.read_text(encoding="ascii"))
        assert router_summary["test_split_included"] is True

        historical_run = Path(args.config).resolve().parent
        assert not (historical_run / "test_predictions.csv").exists()
        assert not (historical_run / "test_metrics.json").exists()
        metric_files = sorted(historical_run.glob("version_*/metrics.csv"))
        assert metric_files
        for metrics_path in metric_files:
            metrics = pd.read_csv(metrics_path)
            test_columns = [column for column in metrics if column.startswith("test_")]
            assert all(metrics[column].notna().sum() == 0 for column in test_columns)
        print(
            "Historical router test artifact: present but unloaded during fit; "
            "benchmark SHA256 matches source"
        )
    else:
        assert not args.allow_present_unloaded_test
    outer_test = normalize(load_csv(source_test_path))
    train_norm = normalize(router_train)
    assert key_set(train_norm).isdisjoint(key_set(outer_test))
    assert set(train_norm["head"]).isdisjoint(set(outer_test["head"]))
    assert set(train_norm["tail"]).isdisjoint(set(outer_test["tail"]))

    for teacher in manifest["teachers"]:
        root = Path(teacher["dataset_dir"]) / "blind_start"
        seen = pd.concat([load_csv(root / "train_set0.csv"), load_csv(root / "val_set0.csv")])
        target = normalize(load_csv(root / "test_set0.csv"))
        assert key_set(seen) | key_set(target) <= key_set(outer_train)
        assert set(normalize(seen)["head"]).isdisjoint(set(target["head"]))
        assert set(normalize(seen)["tail"]).isdisjoint(set(target["tail"]))

    bdb_drugs, bdb_proteins = load_bindingdb_features(bindingdb)
    bdb_fold = load_bindingdb_fold(bindingdb, fold)
    heldout = pd.concat([bdb_fold["val"], bdb_fold["test"]])
    heldout_drugs = {bdb_drugs[x] for x in heldout["head"] if x in bdb_drugs}
    heldout_proteins = {bdb_proteins[x] for x in heldout["tail"] if x in bdb_proteins}
    _, _, metabolic_drugs, metabolic_proteins = load_metabolic_dataset(args.metabolic_dataset_dir)
    assert set(metabolic_drugs.values()).isdisjoint(heldout_drugs)
    assert set(metabolic_proteins.values()).isdisjoint(heldout_proteins)

    source_run = Path(config["experiment"]["save_dir"]) / config["finetune"]["source_run"]
    source_config = yaml.safe_load((source_run / "run_config.yaml").read_text())
    assert f"fold{fold}" in source_config["data"]["dataset_name"]
    assert source_config["evaluation"]["test_after_fit"] is False
    source_dataset = (
        Path(source_config["data"]["base_path"])
        / source_config["data"]["dataset_name"]
    )
    expected_source_dataset = Path(
        f"{Path(args.metabolic_dataset_dir)}_random_pair_split"
    )
    assert source_dataset.resolve() == expected_source_dataset.resolve()
    assert source_config["data"]["set_type"] == "blind_start"
    source_set_id = int(source_config["data"]["set_id"])
    source_split = source_dataset / source_config["data"]["set_type"]
    assert (source_split / f"train_set{source_set_id}.csv").is_file()
    assert (source_split / f"val_set{source_set_id}.csv").is_file()

    diagnostic_summary = source_dataset / "diagnostic_split_summary.json"
    assert diagnostic_summary.is_file()
    diagnostic = json.loads(diagnostic_summary.read_text(encoding="ascii"))
    assert Path(diagnostic["source_dir"]).resolve() == Path(
        args.metabolic_dataset_dir
    ).resolve()
    assert Path(diagnostic["output_dir"]).resolve() == source_dataset.resolve()
    assert diagnostic["mode"] == "random_pair"
    assert diagnostic["train_val_pair_overlap"] == 0
    checkpoints = glob.glob(str(source_run / "best_model_epoch*.ckpt"))
    assert len(checkpoints) == 1, checkpoints
    model, loss_fn = get_model(config), get_loss(config)
    audit_checkpoint_compatibility(checkpoints[0], model, loss_fn, config)
    train_df, val_df, test_df = load_dataframes(config, load_test=False)
    assert test_df is None and len(train_df) == len(router_train)

    print(f"Fold {fold}: router train rows={len(train_df)}, validation rows={len(val_df)}")
    print("Outer-test pair/drug/protein overlap: 0/0/0")
    print("Metabolic held-out structure/sequence overlap: 0/0")
    print(f"Stage-1 checkpoint: {checkpoints[0]}")
    print("V3.7 STAGE-2 FAIL-CLOSED PREFLIGHT OK")


if __name__ == "__main__":
    main()
