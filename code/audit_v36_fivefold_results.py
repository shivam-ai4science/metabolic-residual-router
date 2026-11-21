from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent

PREDICTION_PAIRS = {
    0: (
        ROOT / "fold014_v36_audit/runs/aibind_coldstart_repro_ce_fold0_150ep_20260731_031307/test_predictions.csv",
        ROOT / "fold014_v36_audit/runs/aibind_pathway_gated_fold0_v36_fixed_alpha1_20260731_214726/test_predictions.csv",
    ),
    1: (
        ROOT / "fold014_v36_audit/runs/aibind_coldstart_repro_ce_fold1_150ep_20260731_215452/test_predictions.csv",
        ROOT / "fold014_v36_audit/runs/aibind_pathway_gated_fold1_v36_fixed_alpha1_20260801_132644/test_predictions.csv",
    ),
    2: (
        ROOT / "runs/aibind_coldstart_repro_ce_fold2_150ep_20260801_133407/test_predictions.csv",
        ROOT / "runs/aibind_pathway_gated_fold2_v36_fixed_alpha1_20260802_051810/test_predictions.csv",
    ),
    3: (
        ROOT / "runs/aibind_coldstart_repro_ce_fold3_150ep_20260802_141958/test_predictions.csv",
        ROOT / "runs/aibind_pathway_gated_fold3_v36_fixed_alpha1_20260803_044301/test_predictions.csv",
    ),
    4: (
        ROOT / "fold014_v36_audit/runs/aibind_coldstart_repro_ce_20260726_074952/test_predictions.csv",
        ROOT / "fold014_v36_audit/runs/aibind_pathway_gated_coldstart_from_v36_fixed_alpha1_20260730_170847/test_predictions.csv",
    ),
}


def official_metrics(frame):
    labels = frame["label"].to_numpy()
    probabilities = frame["prob"].to_numpy()
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probabilities),
        "aupr": auc(recall, precision),
        "average_precision": average_precision_score(labels, probabilities),
        "accuracy_0p5": accuracy_score(labels, predictions),
        "f1_0p5": f1_score(labels, predictions),
        "predicted_positive_0p5": int(predictions.sum()),
        "unique_probabilities": int(np.unique(probabilities).size),
    }


def validate_pair(fold, baseline, final):
    key_columns = ["row_index", "head", "tail", "label"]
    if len(baseline) != len(final):
        raise ValueError(f"Fold {fold}: row count mismatch")
    for column in key_columns:
        if column not in baseline or column not in final:
            raise ValueError(f"Fold {fold}: missing {column}")
        if not np.array_equal(baseline[column].to_numpy(), final[column].to_numpy()):
            raise ValueError(f"Fold {fold}: {column} is not aligned")


def confusion_transitions(baseline, final):
    labels = baseline["label"].to_numpy().astype(int)
    base_pred = (baseline["prob"].to_numpy() >= 0.5).astype(int)
    final_pred = (final["prob"].to_numpy() >= 0.5).astype(int)
    return {
        "rescued_fn_0p5": int(((labels == 1) & (base_pred == 0) & (final_pred == 1)).sum()),
        "lost_tp_0p5": int(((labels == 1) & (base_pred == 1) & (final_pred == 0)).sum()),
        "fixed_fp_0p5": int(((labels == 0) & (base_pred == 1) & (final_pred == 0)).sum()),
        "new_fp_0p5": int(((labels == 0) & (base_pred == 0) & (final_pred == 1)).sum()),
    }


def summarize(per_fold):
    rows = []
    for prefix in ("baseline", "final", "delta"):
        for metric in ("auc", "aupr", "average_precision"):
            column = f"{prefix}_{metric}"
            values = per_fold[column].to_numpy(dtype=float)
            rows.append(
                {
                    "quantity": column,
                    "mean": values.mean(),
                    "population_sd": values.std(ddof=0),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "reports/v36_fivefold_final"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, (baseline_path, final_path) in PREDICTION_PAIRS.items():
        baseline = pd.read_csv(baseline_path)
        final = pd.read_csv(final_path)
        validate_pair(fold, baseline, final)
        base_metrics = official_metrics(baseline)
        final_metrics = official_metrics(final)

        row = {"fold": fold, "rows": len(baseline)}
        for name, value in base_metrics.items():
            row[f"baseline_{name}"] = value
        for name, value in final_metrics.items():
            row[f"final_{name}"] = value
        for name in ("auc", "aupr", "average_precision", "accuracy_0p5", "f1_0p5"):
            row[f"delta_{name}"] = final_metrics[name] - base_metrics[name]
        row.update(confusion_transitions(baseline, final))
        rows.append(row)

    per_fold = pd.DataFrame(rows).sort_values("fold")
    summary = summarize(per_fold)
    per_fold.to_csv(output_dir / "tieaware_metrics_by_fold.csv", index=False)
    summary.to_csv(output_dir / "tieaware_metric_summary.csv", index=False)

    display_columns = [
        "fold",
        "baseline_auc",
        "final_auc",
        "delta_auc",
        "baseline_aupr",
        "final_aupr",
        "delta_aupr",
        "rescued_fn_0p5",
        "lost_tp_0p5",
        "fixed_fp_0p5",
        "new_fp_0p5",
    ]
    print(per_fold[display_columns].to_string(index=False))
    print("\nFive-fold mean and population SD:")
    print(summary.to_string(index=False))
    print(f"\nSaved audit tables to {output_dir}")


if __name__ == "__main__":
    main()
