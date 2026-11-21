import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.dataset import get_dataloaders
from src.lightning_module import DPILightningWrapper
from src.losses import get_loss
from src.models import get_model
from utils import coerce_types


torch.autograd.set_detect_anomaly(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set-id", type=int, default=None)
    parser.add_argument("--set-type", default=None)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def split_path(base_path, dataset_name, set_type, split, set_id):
    split_names = [split]
    if split == "val":
        split_names.append("valid")
    candidates = []
    for split_name in split_names:
        filename = f"{split_name}_set{set_id}.csv"
        if set_type:
            candidates.append(os.path.join(base_path, dataset_name, set_type, filename))
        candidates.append(os.path.join(base_path, dataset_name, filename))
        if set_type:
            candidates.append(os.path.join(base_path, set_type, filename))
        candidates.append(os.path.join(base_path, filename))

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {split}_set{set_id}.csv. Tried: {candidates}")


def load_split_df(config, split):
    data = config["data"]
    path = split_path(
        data["base_path"],
        data["dataset_name"],
        data.get("set_type", ""),
        split,
        data["set_id"],
    )
    return pd.read_csv(path, index_col=0)


def normalize_pair_ids(df):
    df = df.copy()
    df["head"] = df["head"].astype(str).str.replace("Drug:", "", regex=False)
    df["tail"] = df["tail"].astype(str).str.replace("Protein:", "", regex=False)
    return df


def filter_missing_features(config, *dfs):
    dataset_name = config["data"]["dataset_name"]
    embedding_name = config["data"].get("embeddings", "")

    drug_exclude = []
    protein_exclude = []
    if dataset_name.startswith("BindingDB_AIBind") and config["data"].get("official_exact", False):
        dataset_path = os.path.join(config["data"]["base_path"], dataset_name)
        exclusion_specs = [
            ("drug_without_feature.txt", "Drug:", drug_exclude),
            ("protein_without_feature.txt", "Protein:", protein_exclude),
        ]
        for filename, prefix, target in exclusion_specs:
            path = os.path.join(dataset_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Official exclusion file not found: {path}")
            with open(path, "r") as handle:
                target.extend(
                    line.split()[0].replace(prefix, "", 1)
                    for line in handle
                    if line.strip()
                )
    elif dataset_name.startswith("BindingDB_AIBind"):
        if embedding_name in {"mol2vec", "coldstartcpi"}:
            drug_exclude = ["141", "303", "836", "1167", "2837", "2957", "3663", "4612", "4723", "5534", "6258"]
            protein_exclude = ["2122", "901", "93"]
        else:
            drug_exclude = ["3631", "1397", "2837", "2957", "4403"]
    elif dataset_name == "esp":
        drug_exclude = ["123", "170"]

    filtered = []
    for df in dfs:
        if df is None:
            filtered.append(None)
            continue
        keep = pd.Series(True, index=df.index)
        if drug_exclude:
            keep = keep & ~df["head"].isin(drug_exclude)
        if protein_exclude:
            keep = keep & ~df["tail"].isin(protein_exclude)
        filtered.append(df[keep].reset_index(drop=True))
    return filtered


def load_dataframes(config, load_test=True):
    train_df = load_split_df(config, "train")
    val_df = load_split_df(config, "val")
    if load_test:
        try:
            test_df = load_split_df(config, "test")
        except FileNotFoundError:
            test_df = None
    else:
        test_df = None

    official_exact = config["data"].get("official_exact", False)
    if official_exact:
        train_fraction = float(config["data"].get("train_fraction", 1.0))
        val_fraction = float(config["data"].get("val_fraction", 1.0))
        if train_fraction != 1.0 or val_fraction != 1.0:
            raise ValueError("official_exact requires full train and validation splits")
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        if test_df is not None:
            test_df = test_df.reset_index(drop=True)
    else:
        train_df = train_df.drop_duplicates().reset_index(drop=True)
        val_df = val_df.drop_duplicates().reset_index(drop=True)
        if test_df is not None:
            test_df = test_df.drop_duplicates().reset_index(drop=True)

        train_df = train_df.sample(frac=config["data"].get("train_fraction", 1.0), random_state=config["trainer"].get("seed", 1234))
        val_df = val_df.sample(frac=config["data"].get("val_fraction", 1.0), random_state=config["trainer"].get("seed", 1234))

    train_df = normalize_pair_ids(train_df)
    val_df = normalize_pair_ids(val_df)
    test_df = normalize_pair_ids(test_df) if test_df is not None else None

    train_df, val_df, test_df = filter_missing_features(config, train_df, val_df, test_df)
    return train_df, val_df, test_df


def inject_dynamic_config(config, train_df, run_dir, run_id):
    pos_count = float(train_df["label"].sum())
    neg_count = float(len(train_df) - pos_count)

    config["experiment"]["run_id"] = run_id
    config["experiment"]["run_dir"] = run_dir

    if config["loss"].get("use_dynamic_weights", True):
        config["loss"].setdefault("params", {})
        config["loss"]["params"]["pos_weight"] = neg_count / max(pos_count, 1.0)

    return config


def save_run_config(config, run_dir):
    with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
        yaml.dump(config, f)


def save_reproducibility_snapshot(config, run_dir, config_path):
    if not config["data"].get("official_exact", False):
        return

    source_files = [
        (config_path, os.path.join("config", os.path.basename(config_path))),
        ("train.py", "train.py"),
        ("src/dataset.py", "src/dataset.py"),
        ("src/lightning_module.py", "src/lightning_module.py"),
        ("src/models/coldstart_cpi_repro.py", "src/models/coldstart_cpi_repro.py"),
        ("src/losses/loss_funk.py", "src/losses/loss_funk.py"),
    ]
    snapshot_dir = os.path.join(run_dir, "source_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    hashes = {}
    for source_path, snapshot_path in source_files:
        normalized_path = os.path.normpath(source_path)
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Reproducibility source file not found: {normalized_path}")
        destination = os.path.join(snapshot_dir, snapshot_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(normalized_path, destination)
        with open(normalized_path, "rb") as handle:
            hashes[snapshot_path] = hashlib.sha256(handle.read()).hexdigest()

    manifest = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "source_sha256": hashes,
    }
    with open(os.path.join(run_dir, "reproducibility_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def audit_checkpoint_compatibility(checkpoint_path, model, loss_fn, config):
    ft_config = config.get("finetune", {})
    if not ft_config.get("validate_checkpoint_keys", False):
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = DPILightningWrapper(model, loss_fn, config).state_dict()

    source_model_keys = [key for key in source_state if key.startswith("model.")]
    matched_keys = [
        key
        for key in source_model_keys
        if key in target_state and tuple(source_state[key].shape) == tuple(target_state[key].shape)
    ]
    mismatched_keys = [
        key
        for key in source_model_keys
        if key in target_state and tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    source_elements = sum(source_state[key].numel() for key in source_model_keys)
    matched_elements = sum(source_state[key].numel() for key in matched_keys)
    coverage = matched_elements / max(source_elements, 1)

    print(
        "Checkpoint compatibility: "
        f"matched={len(matched_keys)}/{len(source_model_keys)} model tensors, "
        f"source_element_coverage={coverage:.6f}"
    )
    if mismatched_keys:
        raise RuntimeError(f"Checkpoint has shape-mismatched model tensors: {mismatched_keys[:10]}")

    minimum_coverage = float(ft_config.get("min_source_model_coverage", 0.99))
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"Checkpoint model coverage {coverage:.6f} is below required {minimum_coverage:.6f}"
        )

    for prefix in ft_config.get("required_source_prefixes", []):
        target_keys = [key for key in target_state if key.startswith(prefix)]
        missing = [key for key in target_keys if key not in matched_keys]
        if not target_keys:
            raise RuntimeError(f"Required checkpoint prefix does not exist in target model: {prefix}")
        if missing:
            raise RuntimeError(f"Checkpoint is missing required tensors for {prefix}: {missing[:10]}")


def build_lightning_model(config, model, loss_fn):
    if config["experiment"].get("mode") != "finetune":
        return DPILightningWrapper(model, loss_fn, config)

    ft_config = config["finetune"]
    source_run_dir = os.path.join(config["experiment"]["save_dir"], ft_config["source_run"])
    if ft_config["checkpoint_type"] == "best":
        if "epoch_number" in ft_config:
            epoch_str = f"{ft_config['epoch_number']:02d}"
            ckpt_path = os.path.join(source_run_dir, f"best_model_epoch{epoch_str}.ckpt")
        else:
            ckpt_path = os.path.join(source_run_dir, "best_model.ckpt")
            if not os.path.exists(ckpt_path):
                best_candidates = sorted(
                    glob.glob(os.path.join(source_run_dir, "best_model_epoch*.ckpt")),
                    key=os.path.getmtime,
                    reverse=True,
                )
                if best_candidates:
                    ckpt_path = best_candidates[0]
    else:
        epoch_str = f"{ft_config['epoch_number']:02d}"
        ckpt_path = os.path.join(source_run_dir, f"snapshot_epoch{epoch_str}.ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    audit_checkpoint_compatibility(ckpt_path, model, loss_fn, config)

    lightning_model = DPILightningWrapper.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        strict=False,
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    if ft_config.get("freeze_backbone", False):
        for name, param in lightning_model.model.named_parameters():
            if any(x in name for x in ["loop_scanner", "helix_scanner", "fold_scanner", "lig_projector"]):
                param.requires_grad = False

    return lightning_model


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = coerce_types(config)

    if args.set_id is not None:
        config["data"]["set_id"] = args.set_id
    if args.set_type is not None:
        config["data"]["set_type"] = args.set_type
    if args.dataset_name is not None:
        config["data"]["dataset_name"] = args.dataset_name

    seed = config["trainer"].get("seed", config["experiment"].get("seed", 1234))
    pl.seed_everything(seed, workers=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['name']}_{run_id}"
    run_dir = os.path.join(config["experiment"]["save_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    test_after_fit = bool(config.get("evaluation", {}).get("test_after_fit", True))
    train_df, val_df, test_df = load_dataframes(config, load_test=test_after_fit)
    if not test_after_fit:
        print("Test split access disabled for this development run.")
    config = inject_dynamic_config(config, train_df, run_dir, run_id)

    if test_df is not None and test_after_fit:
        train_loader, val_loader, test_loader = get_dataloaders(config, train_df, val_df, test_df)
    else:
        train_loader, val_loader = get_dataloaders(config, train_df, val_df)
        test_loader = None

    infer_embedding_dims = config["data"].get("infer_embedding_dims", True)
    if infer_embedding_dims:
        sample_batch = next(iter(train_loader))
        sample_prot, sample_lig = sample_batch[0], sample_batch[1]
        config["model"]["params"]["prot_dim"] = sample_prot.shape[-1]
        config["model"]["params"]["lig_dim"] = sample_lig.shape[-1]
    else:
        required_dims = ["prot_dim", "lig_dim"]
        missing_dims = [key for key in required_dims if key not in config["model"]["params"]]
        if missing_dims:
            raise ValueError(f"Explicit model dimensions required when inference is disabled: {missing_dims}")
    save_run_config(config, run_dir)
    save_reproducibility_snapshot(config, run_dir, args.config)

    model = get_model(config)
    loss_fn = get_loss(config)
    lightning_model = build_lightning_model(config, model, loss_fn)

    monitor_metric = config["trainer"].get("monitor_metric", "val_score")
    monitor_mode = config["trainer"].get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename="best_model_epoch{epoch:02d}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            auto_insert_metric_name=False,
        )
    ]

    save_interval = config["trainer"].get("save_every_n_epochs", 0)
    if save_interval > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                filename="snapshot_epoch{epoch:02d}",
                every_n_epochs=save_interval,
                save_top_k=-1,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
            )
        )

    patience = config["trainer"].get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=int(patience)))

    trainer_config = config["trainer"]
    gradient_clip_val = float(trainer_config.get("gradient_clip_val", 1.0))
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="auto",
        devices=1,
        logger=CSVLogger(save_dir=config["experiment"]["save_dir"], name=run_name),
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=trainer_config.get("gradient_clip_algorithm", "norm"),
        precision="32-true",
        accumulate_grad_batches=config["trainer"].get("accumulation_steps", 1),
        deterministic=trainer_config.get("deterministic", True),
        num_sanity_val_steps=int(trainer_config.get("num_sanity_val_steps", 2)),
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    if test_loader is not None:
        trainer.test(lightning_model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
