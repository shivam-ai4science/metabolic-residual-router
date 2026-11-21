from pathlib import Path
import argparse

import pandas as pd
import torch
import yaml

from src.dataset import OfficialFixedPaddingCollate, get_dataloaders
from src.models import get_model
from train import load_dataframes, split_path
from utils import coerce_types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_aibind_coldstart_official_exact_fold4.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_exclusions(dataset_dir, filename):
    return {
        line.split()[0]
        for line in (dataset_dir / filename).read_text().splitlines()
        if line.strip()
    }


def expected_official_frame(config, split):
    data = config["data"]
    path = split_path(
        data["base_path"],
        data["dataset_name"],
        data["set_type"],
        split,
        data["set_id"],
    )
    frame = pd.read_csv(path)[["head", "tail", "label"]]
    dataset_dir = Path(data["base_path"]) / data["dataset_name"]
    excluded_drugs = read_exclusions(dataset_dir, "drug_without_feature.txt")
    excluded_proteins = read_exclusions(dataset_dir, "protein_without_feature.txt")
    frame = frame[
        ~frame["head"].isin(excluded_drugs)
        & ~frame["tail"].isin(excluded_proteins)
    ].reset_index(drop=True)
    frame["head"] = frame["head"].str.replace("Drug:", "", regex=False)
    frame["tail"] = frame["tail"].str.replace("Protein:", "", regex=False)
    return frame


def assert_config(config):
    data = config["data"]
    trainer = config["trainer"]
    params = config["model"]["params"]
    expected = {
        "data.official_exact": data.get("official_exact"),
        "data.padding_mode": data.get("padding_mode"),
        "data.infer_embedding_dims": data.get("infer_embedding_dims"),
        "data.batch_size_train": data.get("batch_size_train"),
        "data.batch_size_val": data.get("batch_size_val"),
        "data.official_lig_local_tokens": data.get("official_lig_local_tokens"),
        "data.official_prot_local_tokens": data.get("official_prot_local_tokens"),
        "model.prot_dim": params.get("prot_dim"),
        "model.lig_dim": params.get("lig_dim"),
        "model.latent_dim": params.get("latent_dim"),
        "model.num_heads": params.get("num_heads"),
        "model.dropout": params.get("dropout"),
        "model.use_padding_mask": params.get("use_padding_mask"),
        "model.output_dim": params.get("output_dim"),
        "trainer.learning_rate": trainer.get("learning_rate"),
        "trainer.weight_decay": trainer.get("weight_decay"),
        "trainer.optimizer": trainer.get("optimizer"),
        "trainer.monitor_metric": trainer.get("monitor_metric"),
        "trainer.max_epochs": trainer.get("max_epochs"),
        "trainer.early_stopping_patience": trainer.get("early_stopping_patience"),
        "trainer.gradient_clip_val": trainer.get("gradient_clip_val"),
        "trainer.deterministic": trainer.get("deterministic"),
        "trainer.num_sanity_val_steps": trainer.get("num_sanity_val_steps"),
    }
    required = {
        "data.official_exact": True,
        "data.padding_mode": "official_fixed",
        "data.infer_embedding_dims": False,
        "data.batch_size_train": 256,
        "data.batch_size_val": 256,
        "data.official_lig_local_tokens": 100,
        "data.official_prot_local_tokens": 1000,
        "model.prot_dim": 1024,
        "model.lig_dim": 300,
        "model.latent_dim": 512,
        "model.num_heads": 4,
        "model.dropout": 0.1,
        "model.use_padding_mask": False,
        "model.output_dim": 2,
        "trainer.learning_rate": 1e-4,
        "trainer.weight_decay": 0.0,
        "trainer.optimizer": "Adam",
        "trainer.monitor_metric": "val_score",
        "trainer.max_epochs": 500,
        "trainer.early_stopping_patience": 50,
        "trainer.gradient_clip_val": 0.0,
        "trainer.deterministic": False,
        "trainer.num_sanity_val_steps": 0,
    }
    mismatches = {key: (expected[key], value) for key, value in required.items() if expected[key] != value}
    if mismatches:
        raise AssertionError(f"Official-exact config mismatch: {mismatches}")


def main():
    args = parse_args()
    with open(args.config, "r") as handle:
        config = coerce_types(yaml.safe_load(handle))
    assert_config(config)

    train_df, val_df, test_df = load_dataframes(config)
    loaded = {"train": train_df, "val": val_df, "test": test_df}
    expected_stats = {
        "train": (18370, 15),
        "val": (23897, 5),
        "test": (7932, 2),
    }
    for split, frame in loaded.items():
        expected = expected_official_frame(config, split)
        pd.testing.assert_frame_equal(
            frame[["head", "tail", "label"]].reset_index(drop=True),
            expected,
            check_dtype=False,
        )
        duplicate_count = int(frame[["head", "tail", "label"]].duplicated().sum())
        if (len(frame), duplicate_count) != expected_stats[split]:
            raise AssertionError(
                f"Unexpected Fold 4 {split} rows/duplicates: "
                f"{(len(frame), duplicate_count)} != {expected_stats[split]}"
            )
        print(
            f"{split}: rows={len(frame)} "
            f"duplicates_preserved={duplicate_count}"
        )

    train_loader, _, _ = get_dataloaders(config, train_df, val_df, test_df)
    collate = OfficialFixedPaddingCollate(
        prot_local_tokens=config["data"]["official_prot_local_tokens"],
        lig_local_tokens=config["data"]["official_lig_local_tokens"],
    )
    first_item = train_loader.dataset[0]
    second_item = train_loader.dataset[1]
    single = collate([first_item])
    paired = collate([first_item, second_item])

    assert tuple(single[0].shape) == (1, 1001, 1024)
    assert tuple(single[1].shape) == (1, 101, 300)
    assert tuple(paired[0].shape) == (2, 1001, 1024)
    assert tuple(paired[1].shape) == (2, 101, 300)
    print("fixed shapes: protein=(B,1001,1024) ligand=(B,101,300)")

    torch.manual_seed(int(config["trainer"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["trainer"]["seed"]))
    device = torch.device(args.device)
    model = get_model(config).to(device).eval()

    def logits(batch):
        with torch.no_grad():
            output = model(
                batch[0].to(device),
                batch[1].to(device),
                batch[3].to(device),
                batch[2].to(device),
            )
        return output["logits"].detach().cpu()

    single_logits = logits(single)[0]
    paired_logits = logits(paired)[0]
    torch.testing.assert_close(single_logits, paired_logits, rtol=1e-5, atol=1e-6)
    print(f"batch-invariant first-row logits: {single_logits.tolist()}")
    print("OFFICIAL-EXACT PREFLIGHT OK")


if __name__ == "__main__":
    main()
