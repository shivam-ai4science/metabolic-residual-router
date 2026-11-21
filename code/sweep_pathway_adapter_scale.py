import argparse
import glob
import os

import pandas as pd
import torch
import yaml

from src.dataset import get_dataloaders
from src.lightning_module import DPILightningWrapper, pr_auc, roc_auc
from src.losses import get_loss
from src.models import get_model
from train import load_dataframes
from utils import coerce_types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Target data/model config to evaluate.")
    parser.add_argument("--source-run", required=True, help="Run folder containing the metabolic adapter checkpoint.")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alphas", default="-1,-0.5,-0.25,0,0.05,0.1,0.2,0.35,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_checkpoint(run_dir, checkpoint):
    if checkpoint != "best":
        if os.path.isabs(checkpoint) or os.path.exists(checkpoint):
            return checkpoint
        return os.path.join(run_dir, checkpoint)

    matches = sorted(glob.glob(os.path.join(run_dir, "best_model_epoch*.ckpt")), key=os.path.getmtime)
    if not matches:
        matches = sorted(glob.glob(os.path.join(run_dir, "*.ckpt")), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return matches[-1]


def choose_df(split, train_df, val_df, test_df):
    if split == "train":
        return train_df
    if split == "val":
        return val_df
    if test_df is None:
        raise FileNotFoundError("No test split exists for this config.")
    return test_df


def positive_probs(logits):
    if logits.ndim == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.reshape(-1))


def add_scaled_correction(base_logits, correction, alpha):
    if base_logits.ndim == 2 and base_logits.size(1) == 2:
        correction_logits = torch.stack([-0.5 * alpha * correction, 0.5 * alpha * correction], dim=1)
        return base_logits + correction_logits
    return base_logits.reshape(-1) + alpha * correction


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = coerce_types(yaml.safe_load(f))

    config.setdefault("evaluation", {})
    config["evaluation"]["export_predictions"] = False
    config["experiment"]["mode"] = "finetune"
    config.setdefault("finetune", {})
    config["finetune"]["source_run"] = os.path.basename(args.source_run.rstrip("/"))
    config["finetune"]["checkpoint_type"] = "best"
    config["data"]["batch_size_train"] = args.batch_size
    config["data"]["batch_size_val"] = args.batch_size

    train_df, val_df, test_df = load_dataframes(config)
    target_df = choose_df(args.split, train_df, val_df, test_df)
    _, loader = get_dataloaders(config, target_df, target_df)

    sample_batch = next(iter(loader))
    config["model"]["params"]["prot_dim"] = sample_batch[0].shape[-1]
    config["model"]["params"]["lig_dim"] = sample_batch[1].shape[-1]

    model = get_model(config)
    loss_fn = get_loss(config)
    checkpoint_path = resolve_checkpoint(args.source_run, args.checkpoint)
    lightning_model = DPILightningWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    lightning_model.to(device)
    lightning_model.eval()

    labels_all = []
    base_probs_all = []
    corrections_all = []
    gates_all = []
    base_logits_all = []

    with torch.no_grad():
        for batch in loader:
            prot, lig, prot_mask, lig_mask, labels, weights, metas = lightning_model._unpack_batch(batch)
            prot = prot.to(device)
            lig = lig.to(device)
            prot_mask = prot_mask.to(device) if prot_mask is not None else None
            lig_mask = lig_mask.to(device) if lig_mask is not None else None

            outputs = lightning_model(prot, lig, lig_mask, prot_mask)
            base_logits = outputs["base_logits"].detach().cpu()
            correction = outputs["pathway_correction"].detach().cpu()

            labels_all.append(labels.cpu().long())
            base_logits_all.append(base_logits)
            base_probs_all.append(positive_probs(base_logits))
            corrections_all.append(correction)
            gates_all.append(outputs["pathway_gate"].detach().cpu())

    labels = torch.cat(labels_all).numpy()
    base_logits = torch.cat(base_logits_all)
    base_probs = torch.cat(base_probs_all).numpy()
    corrections = torch.cat(corrections_all)
    gates = torch.cat(gates_all)

    rows = []
    for alpha in [float(x.strip()) for x in args.alphas.split(",") if x.strip()]:
        scaled_logits = add_scaled_correction(base_logits, corrections, alpha)
        probs = positive_probs(scaled_logits).numpy()
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc(labels, probs)
        aupr = pr_auc(labels, probs)
        rows.append(
            {
                "alpha": alpha,
                "auc": auc,
                "aupr": aupr,
                "score": auc + aupr,
                "pred_pos": int(preds.sum()),
                "tp": int(((labels == 1) & (preds == 1)).sum()),
                "fp": int(((labels == 0) & (preds == 1)).sum()),
                "mean_abs_scaled_correction": float((alpha * corrections).abs().mean()),
                "mean_scaled_correction": float((alpha * corrections).mean()),
            }
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary = {
        "rows": len(labels),
        "base_auc": roc_auc(labels, base_probs),
        "base_aupr": pr_auc(labels, base_probs),
        "base_score": roc_auc(labels, base_probs) + pr_auc(labels, base_probs),
        "correction_auc": roc_auc(labels, corrections.numpy()),
        "correction_aupr": pr_auc(labels, corrections.numpy()),
        "mean_gate": float(gates.mean()),
        "mean_abs_correction": float(corrections.abs().mean()),
        "mean_correction": float(corrections.mean()),
    }

    output = args.output or os.path.join(
        args.source_run,
        f"{args.split}_bindingdb_pathway_scale_sweep.csv",
    )
    df.to_csv(output, index=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved: {output}")
    print(pd.DataFrame([summary]).to_string(index=False))
    print()
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
