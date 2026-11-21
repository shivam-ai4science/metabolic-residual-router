import argparse
import glob
import os

import pandas as pd
import torch
import yaml

from src.dataset import get_dataloaders
from src.losses import get_loss
from src.models import get_model
from src.lightning_module import DPILightningWrapper
from train import load_dataframes
from utils import coerce_types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def resolve_checkpoint(run_dir, checkpoint):
    if checkpoint != "best":
        if os.path.isabs(checkpoint) or os.path.exists(checkpoint):
            return checkpoint
        return os.path.join(run_dir, checkpoint)

    matches = sorted(
        glob.glob(os.path.join(run_dir, "best_model_epoch*.ckpt")),
        key=os.path.getmtime,
    )
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
        raise FileNotFoundError("No test split exists for this run config.")
    return test_df


def probability_from_logits(raw_logits):
    if raw_logits.ndim == 2 and raw_logits.size(1) == 2:
        probs = torch.softmax(raw_logits, dim=1)[:, 1]
        preds = torch.argmax(raw_logits, dim=1)
        logits = raw_logits[:, 1] - raw_logits[:, 0]
    else:
        logits = raw_logits.reshape(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    return logits.cpu().numpy(), probs.cpu().numpy(), preds.cpu().numpy().astype(int)


def main():
    args = parse_args()
    config_path = os.path.join(args.run_dir, "run_config.yaml")
    with open(config_path, "r") as f:
        config = coerce_types(yaml.safe_load(f))

    config.setdefault("evaluation", {})
    config["evaluation"]["export_predictions"] = False
    config["experiment"]["run_dir"] = args.run_dir
    if args.batch_size is not None:
        config["data"]["batch_size_train"] = args.batch_size
        config["data"]["batch_size_val"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    train_df, val_df, test_df = load_dataframes(config)
    target_df = choose_df(args.split, train_df, val_df, test_df)

    _, loader = get_dataloaders(config, target_df, target_df)
    sample_batch = next(iter(loader))
    config["model"]["params"]["prot_dim"] = sample_batch[0].shape[-1]
    config["model"]["params"]["lig_dim"] = sample_batch[1].shape[-1]

    model = get_model(config)
    loss_fn = get_loss(config)
    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
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

    rows = []
    with torch.no_grad():
        for batch in loader:
            prot, lig, prot_mask, lig_mask, labels, weights, metas = lightning_model._unpack_batch(batch)
            prot = prot.to(device)
            lig = lig.to(device)
            prot_mask = prot_mask.to(device) if prot_mask is not None else None
            lig_mask = lig_mask.to(device) if lig_mask is not None else None
            outputs = lightning_model(prot, lig, lig_mask, prot_mask)
            raw_logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logits, probs, preds = probability_from_logits(raw_logits.detach())
            labels_np = labels.cpu().numpy().astype(int)
            for i, label in enumerate(labels_np):
                meta = metas[i] if i < len(metas) else {}
                rows.append(
                    {
                        "row_index": meta.get("row_index", i),
                        "head": meta.get("head", ""),
                        "tail": meta.get("tail", ""),
                        "label": int(label),
                        "logit": float(logits[i]),
                        "prob": float(probs[i]),
                        "pred": int(preds[i]),
                    }
                )

    output = args.output or os.path.join(args.run_dir, f"{args.split}_predictions.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved {len(rows)} {args.split} predictions: {output}")


if __name__ == "__main__":
    main()
