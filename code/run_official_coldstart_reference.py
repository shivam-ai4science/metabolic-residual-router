import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from dataset import load_BindingDB_AIBind_dataset
from model import ColdstartCPI


OFFICIAL_SOURCE_SHA256 = {
    "dataset.py": "ed24257a30b58fd2c52e7d7018eb69c32b8e350887d97a90cafff5fd2ebb3f23",
    "model.py": "864657d4f644a3e5cbf062c890ef840b0536a3a9ece250596fcfbceab59c5862",
    "train_BindingDB_AIBind.py": "ab3dc12acb4f76c8270701547f1a404b0033f057fef2472d9e99cab0c3e83cf6",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auditable reference runner for the official ColdstartCPI implementation."
    )
    parser.add_argument("--scenarios", default="blind_start")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def roc_auc(labels, scores):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    return float(metrics.auc(fpr, tpr))


def pr_auc(labels, scores):
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    return float(metrics.auc(recall, precision))


def evaluate(model, loader, loss_fn, description, keep_scores=False):
    model.eval()
    losses = []
    labels_all = []
    predictions_all = []
    scores_all = []
    progress = tqdm(
        enumerate(BackgroundGenerator(loader)),
        total=len(loader),
        desc=description,
    )
    with torch.no_grad():
        for _, data in progress:
            input_batch, labels = data
            labels = labels.cuda()
            input_batch = [tensor.cuda() for tensor in input_batch]
            logits = model(input_batch)
            loss = loss_fn(logits, labels)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())
            scores_all.extend(probabilities[:, 1].cpu().numpy())
            losses.append(loss.item())

    result = {
        "loss": float(np.average(losses)),
        "acc": float(accuracy_score(labels_all, predictions_all)),
        "precision": float(precision_score(labels_all, predictions_all, zero_division=0)),
        "recall": float(recall_score(labels_all, predictions_all, zero_division=0)),
        "f1": float(f1_score(labels_all, predictions_all, zero_division=0)),
        "auc": roc_auc(labels_all, scores_all),
        "aupr": pr_auc(labels_all, scores_all),
    }
    if keep_scores:
        result["labels"] = [int(value) for value in labels_all]
        result["predictions"] = [int(value) for value in predictions_all]
        result["scores"] = [float(value) for value in scores_all]
    return result


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(output_dir, script_dir, args):
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_names = [
        "run_official_coldstart_reference.py",
        "train_BindingDB_AIBind.py",
        "dataset.py",
        "model.py",
    ]
    hashes = {}
    for name in source_names:
        source = script_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Required official source file missing: {source}")
        shutil.copy2(source, snapshot_dir / name)
        hashes[name] = sha256(source)
        expected_hash = OFFICIAL_SOURCE_SHA256.get(name)
        if expected_hash is not None and hashes[name] != expected_hash:
            raise RuntimeError(
                f"Official source hash mismatch for {name}: "
                f"{hashes[name]} != {expected_hash}"
            )

    manifest = {
        "arguments": vars(args),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "source_sha256": hashes,
        "notes": [
            "Official model.py and dataset.py are imported directly.",
            "The seed is set once before the requested fold sequence, matching the official loop.",
            "A standalone fold greater than zero does not reproduce the official cross-fold RNG trajectory.",
        ],
    }
    (output_dir / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def save_test_predictions(path, loader, result):
    pairs = loader.dataset.pairs
    if len(pairs) != len(result["labels"]):
        raise AssertionError("Prediction count does not match the official filtered test set")

    with path.open("w", newline="") as handle:
        fieldnames = ["row_index", "head", "tail", "label", "prob", "pred"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (pair, label, score, prediction) in enumerate(
            zip(pairs, result["labels"], result["scores"], result["predictions"])
        ):
            if int(float(pair[2])) != label:
                raise AssertionError(f"Label/order mismatch at filtered test row {index}")
            writer.writerow(
                {
                    "row_index": index,
                    "head": pair[0],
                    "tail": pair[1],
                    "label": label,
                    "prob": score,
                    "pred": prediction,
                }
            )


def validate_args(args):
    if sorted(set(args.folds)) != args.folds:
        raise ValueError("--folds must be unique and provided in ascending order")
    if any(fold < 0 or fold > 4 for fold in args.folds):
        raise ValueError("Fold ids must be between 0 and 4")
    if args.folds[0] != 0:
        raise ValueError(
            "An official RNG-trajectory run must begin with fold 0. "
            "Use --folds 0 for the first anchor or --folds 0 1 2 3 4 for the full reference."
        )


def main():
    args = parse_args()
    validate_args(args)
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    if args.output_root is None:
        output_root = script_dir.parents[2] / "DPI_experiments" / "runs"
    else:
        output_root = Path(args.output_root).expanduser().resolve()
    fold_tag = "-".join(str(fold) for fold in args.folds)
    run_name = f"official_coldstart_reference_folds{fold_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    save_manifest(output_dir, script_dir, args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    loss_fn = nn.CrossEntropyLoss(weight=None)
    fold_results = []
    for fold in args.folds:
        print(f"{'*' * 24} Fold {fold} {'*' * 24}", flush=True)
        train_loader, val_loader, test_loader = load_BindingDB_AIBind_dataset(
            "BindingDB_AIBind",
            scenarios=args.scenarios,
            batch_size=args.batch_size,
            fold=fold,
        )
        model = ColdstartCPI(unify_num=512, head_num=4).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        checkpoint = output_dir / f"valid_best_checkpoint{fold}.pth"
        epoch_metrics = output_dir / f"fold{fold}_epoch_metrics.csv"
        best_score = 0.0
        best_epoch = 0
        wait = 0

        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            progress = tqdm(
                enumerate(BackgroundGenerator(train_loader)),
                total=len(train_loader),
                desc=f"fold{fold} train epoch{epoch}",
            )
            for _, train_data in progress:
                input_batch, labels = train_data
                input_batch = [tensor.cuda() for tensor in input_batch]
                labels = labels.cuda()
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = loss_fn(logits, labels)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            validation = evaluate(
                model,
                val_loader,
                loss_fn,
                description=f"fold{fold} val epoch{epoch}",
            )
            validation_score = validation["auc"] + validation["aupr"]
            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.average(train_losses)),
                "val_loss": validation["loss"],
                "val_auc": validation["auc"],
                "val_aupr": validation["aupr"],
                "val_score": validation_score,
                "patience_before_update": wait,
            }
            append_csv(epoch_metrics, row)
            print(json.dumps(row, sort_keys=True), flush=True)

            if validation_score > best_score:
                best_score = validation_score
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                wait += 1
            if wait == args.patience:
                print(f"Fold {fold} early stop at epoch {epoch}", flush=True)
                break

        model.load_state_dict(torch.load(checkpoint))

        # The official script evaluates the shuffled train loader before test.
        # Keeping this pass preserves the RNG trajectory for subsequent folds.
        train_result = evaluate(
            model,
            train_loader,
            loss_fn,
            description=f"fold{fold} final train",
        )
        test_result = evaluate(
            model,
            test_loader,
            loss_fn,
            description=f"fold{fold} final test",
            keep_scores=True,
        )
        save_test_predictions(output_dir / f"fold{fold}_test_predictions.csv", test_loader, test_result)

        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            **{f"train_{key}": value for key, value in train_result.items()},
            **{
                f"test_{key}": value
                for key, value in test_result.items()
                if key not in {"labels", "predictions", "scores"}
            },
        }
        fold_results.append(fold_result)
        append_csv(output_dir / "fold_results.csv", fold_result)
        print(json.dumps(fold_result, indent=2, sort_keys=True), flush=True)

    summary = {"run_dir": str(output_dir), "fold_results": fold_results}
    if len(fold_results) > 1:
        for metric_name in ["test_auc", "test_aupr"]:
            values = np.asarray([row[metric_name] for row in fold_results], dtype=float)
            summary[f"mean_{metric_name}"] = float(values.mean())
            summary[f"population_sd_{metric_name}"] = float(np.sqrt(values.var()))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"OFFICIAL REFERENCE COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
