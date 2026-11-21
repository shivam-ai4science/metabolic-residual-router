import argparse
import glob
import os

import pytorch_lightning as pl
import yaml

from src.dataset import get_dataloaders
from src.lightning_module import DPILightningWrapper
from src.losses import get_loss
from src.models import get_model
from train import load_dataframes
from utils import coerce_types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--set-id", type=int, default=None)
    parser.add_argument("--set-type", default=None)
    return parser.parse_args()


def resolve_checkpoint(run_dir, checkpoint):
    if checkpoint != "best":
        return checkpoint if os.path.isabs(checkpoint) else os.path.join(run_dir, checkpoint)

    matches = sorted(glob.glob(os.path.join(run_dir, "best_model_epoch*.ckpt")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(run_dir, "*.ckpt")))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return matches[-1]


def main():
    args = parse_args()
    config_path = os.path.join(args.run_dir, "run_config.yaml")
    with open(config_path, "r") as f:
        config = coerce_types(yaml.safe_load(f))

    if args.set_id is not None:
        config["data"]["set_id"] = args.set_id
    if args.set_type is not None:
        config["data"]["set_type"] = args.set_type

    config.setdefault("evaluation", {})
    config["evaluation"]["export_predictions"] = True
    config["experiment"]["run_dir"] = args.run_dir

    _, _, test_df = load_dataframes(config)
    if test_df is None:
        raise FileNotFoundError("No test split found for this run config.")

    train_loader, val_loader, test_loader = get_dataloaders(config, test_df, test_df, test_df)
    sample_batch = next(iter(test_loader))
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

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, precision="32-true")
    trainer.test(lightning_model, test_loader)


if __name__ == "__main__":
    main()
