import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit how a residual model repairs or breaks baseline rankings."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def binary_log_loss(labels, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def validate(frame):
    required = {
        "label",
        "base_logit",
        "base_prob",
        "final_logit",
        "final_prob",
        "correction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    labels = frame["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1 values")
    for column in required.difference({"label"}):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column contains non-finite values: {column}")


def pairwise_audit(labels, base_scores, final_scores, chunk_size):
    positive_base = base_scores[labels == 1]
    negative_base = base_scores[labels == 0]
    positive_final = final_scores[labels == 1]
    negative_final = final_scores[labels == 0]

    gap_edges = np.array(
        [-np.inf, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
         0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf],
        dtype=float,
    )
    gap_pairs = np.zeros(len(gap_edges) - 1, dtype=np.int64)
    gap_repairs = np.zeros_like(gap_pairs)
    gap_breaks = np.zeros_like(gap_pairs)

    repaired = 0
    broken = 0
    base_correct = 0
    base_inverted = 0
    base_tied = 0
    final_tied = 0

    for start in range(0, len(positive_base), chunk_size):
        stop = min(start + chunk_size, len(positive_base))
        base_gap = positive_base[start:stop, None] - negative_base[None, :]
        final_gap = positive_final[start:stop, None] - negative_final[None, :]

        base_ok = base_gap > 0.0
        base_bad = base_gap < 0.0
        final_ok = final_gap > 0.0
        final_bad = final_gap < 0.0
        repair = base_bad & final_ok
        damage = base_ok & final_bad

        repaired += int(repair.sum())
        broken += int(damage.sum())
        base_correct += int(base_ok.sum())
        base_inverted += int(base_bad.sum())
        base_tied += int((base_gap == 0.0).sum())
        final_tied += int((final_gap == 0.0).sum())

        band = np.searchsorted(gap_edges, base_gap.ravel(), side="right") - 1
        band = np.clip(band, 0, len(gap_pairs) - 1)
        gap_pairs += np.bincount(band, minlength=len(gap_pairs))
        gap_repairs += np.bincount(
            band, weights=repair.ravel().astype(np.int64), minlength=len(gap_pairs)
        ).astype(np.int64)
        gap_breaks += np.bincount(
            band, weights=damage.ravel().astype(np.int64), minlength=len(gap_pairs)
        ).astype(np.int64)

    changed = repaired + broken
    summary = {
        "positive_negative_pairs": int(len(positive_base) * len(negative_base)),
        "base_correct_pairs": base_correct,
        "base_inverted_pairs": base_inverted,
        "base_tied_pairs": base_tied,
        "final_tied_pairs": final_tied,
        "repaired_pairs": repaired,
        "broken_pairs": broken,
        "net_repaired_pairs": repaired - broken,
        "changed_order_precision": repaired / changed if changed else None,
        "fraction_base_inversions_repaired": repaired / base_inverted if base_inverted else None,
        "fraction_base_correct_pairs_broken": broken / base_correct if base_correct else None,
    }

    labels_out = []
    for lower, upper in zip(gap_edges[:-1], gap_edges[1:]):
        labels_out.append(f"{lower:g}..{upper:g}")
    gap_frame = pd.DataFrame(
        {
            "base_pair_gap": labels_out,
            "pairs": gap_pairs,
            "repairs": gap_repairs,
            "breaks": gap_breaks,
        }
    )
    gap_frame["net_repairs"] = gap_frame["repairs"] - gap_frame["breaks"]
    changed_by_gap = gap_frame["repairs"] + gap_frame["breaks"]
    gap_frame["changed_order_precision"] = np.where(
        changed_by_gap > 0,
        gap_frame["repairs"] / changed_by_gap,
        np.nan,
    )
    return summary, gap_frame


def main():
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    validate(frame)

    labels = frame["label"].to_numpy(dtype=int)
    base_logits = frame["base_logit"].to_numpy(dtype=float)
    final_logits = frame["final_logit"].to_numpy(dtype=float)
    base_prob = frame["base_prob"].to_numpy(dtype=float)
    final_prob = frame["final_prob"].to_numpy(dtype=float)
    correction = frame["correction"].to_numpy(dtype=float)

    base_auc = float(roc_auc_score(labels, base_logits))
    final_auc = float(roc_auc_score(labels, final_logits))
    base_aupr = pr_auc(labels, base_logits)
    final_aupr = pr_auc(labels, final_logits)
    correction_auc = float(roc_auc_score(labels, correction))
    correction_aupr = pr_auc(labels, correction)

    pair_summary, gap_frame = pairwise_audit(
        labels,
        base_logits,
        final_logits,
        max(int(args.chunk_size), 1),
    )

    base_pred = base_prob >= 0.5
    final_pred = final_prob >= 0.5
    aligned = np.where(labels == 1, correction > 0.0, correction < 0.0)
    summary = {
        "predictions": str(Path(args.predictions).resolve()),
        "rows": int(len(frame)),
        "positive_rows": int(labels.sum()),
        "negative_rows": int((labels == 0).sum()),
        "base_auc": base_auc,
        "final_auc": final_auc,
        "delta_auc": final_auc - base_auc,
        "base_aupr": base_aupr,
        "final_aupr": final_aupr,
        "delta_aupr": final_aupr - base_aupr,
        "base_score": base_auc + base_aupr,
        "final_score": final_auc + final_aupr,
        "delta_score": (final_auc + final_aupr) - (base_auc + base_aupr),
        "base_log_loss": binary_log_loss(labels, base_prob),
        "final_log_loss": binary_log_loss(labels, final_prob),
        "correction_auc": correction_auc,
        "correction_aupr": correction_aupr,
        "mean_correction": float(correction.mean()),
        "mean_abs_correction": float(np.abs(correction).mean()),
        "positive_mean_correction": float(correction[labels == 1].mean()),
        "negative_mean_correction": float(correction[labels == 0].mean()),
        "label_aligned_correction_fraction": float(aligned.mean()),
        "rescued_fn_0p5": int(((labels == 1) & ~base_pred & final_pred).sum()),
        "lost_tp_0p5": int(((labels == 1) & base_pred & ~final_pred).sum()),
        "fixed_fp_0p5": int(((labels == 0) & base_pred & ~final_pred).sum()),
        "new_fp_0p5": int(((labels == 0) & ~base_pred & final_pred).sum()),
        **pair_summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "incremental_rank_repair_summary.json"
    gap_path = output_dir / "incremental_rank_repairs_by_base_gap.csv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    gap_frame.to_csv(gap_path, index=False)

    print(json.dumps(summary, indent=2))
    print("\nPAIR-GAP BREAKDOWN (changed orders only)")
    changed = gap_frame[(gap_frame["repairs"] + gap_frame["breaks"]) > 0]
    print(changed.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {gap_path}")


if __name__ == "__main__":
    main()
