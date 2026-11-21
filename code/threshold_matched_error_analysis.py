import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def strip_prefix(value, prefix):
    return str(value).replace(f"{prefix}:", "")


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def motif_bucket(smiles):
    s = str(smiles)
    upper = s.upper()
    c_count = upper.count("C")
    o_count = upper.count("O")
    n_count = upper.count("N")
    if any(token in upper for token in ["[ZN", "[MG", "[MN", "[FE", "[CA", "[NA", "[K]", "[CL-", "[CU"]):
        return "metal_inorganic"
    if "P(=O)" in upper or "P(O)" in upper or "[P" in upper:
        if n_count >= 4 or "N1" in upper:
            return "nucleotide_cofactor"
        return "phosphate"
    if c_count <= 2 and o_count >= 2:
        return "small_simple"
    if o_count >= 4 and c_count >= 3:
        return "sugar_polyol"
    if ("C(=O)O" in upper or "C(O)=O" in upper) and n_count >= 1:
        return "aa_carboxylate"
    if "C(=O)O" in upper or "C(O)=O" in upper:
        return "carboxylate"
    return "other"


def metrics(label, pred):
    tp = int(((label == 1) & (pred == 1)).sum())
    fn = int(((label == 1) & (pred == 0)).sum())
    fp = int(((label == 0) & (pred == 1)).sum())
    tn = int(((label == 0) & (pred == 0)).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pred_pos": tp + fp,
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def pick_threshold_for_pred_pos(probs, target_pred_pos):
    sorted_probs = np.sort(probs)[::-1]
    if target_pred_pos <= 0:
        return 1.0 + 1e-9
    if target_pred_pos >= len(sorted_probs):
        return -1e-9
    return float((sorted_probs[target_pred_pos - 1] + sorted_probs[target_pred_pos]) / 2.0)


def pick_threshold_for_fpr(labels, probs, target_fpr):
    thresholds = np.unique(np.r_[0.0, probs, 1.0])[::-1]
    best = None
    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        m = metrics(labels, pred)
        diff = abs(m["fpr"] - target_fpr)
        candidate = (diff, -m["recall"], threshold, m)
        if best is None or candidate < best:
            best = candidate
    return float(best[2])


def add_predictions(df, threshold):
    df = df.copy()
    df["candidate_pred_matched"] = (df["candidate_prob"] >= threshold).astype(int)
    df["transition_matched"] = (
        df["label"].astype(str)
        + ":"
        + df["baseline_pred"].astype(str)
        + "->"
        + df["candidate_pred_matched"].astype(str)
    )
    return df


def transition_counts(df, candidate_pred_col):
    label = df["label"].astype(int)
    base = df["baseline_pred"].astype(int)
    cand = df[candidate_pred_col].astype(int)
    return {
        "rescued_fns": int(((label == 1) & (base == 0) & (cand == 1)).sum()),
        "lost_tps": int(((label == 1) & (base == 1) & (cand == 0)).sum()),
        "fixed_fps": int(((label == 0) & (base == 1) & (cand == 0)).sum()),
        "new_fps": int(((label == 0) & (base == 0) & (cand == 1)).sum()),
    }


def bucket_summary(df, candidate_pred_col):
    rows = []
    for bucket, group in df.groupby("motif_bucket", dropna=False):
        base_m = metrics(group["label"].astype(int).values, group["baseline_pred"].astype(int).values)
        cand_m = metrics(group["label"].astype(int).values, group[candidate_pred_col].astype(int).values)
        trans = transition_counts(group, candidate_pred_col)
        row = {"motif_bucket": bucket, "rows": len(group), "pos": int((group["label"] == 1).sum())}
        row.update({f"baseline_{k}": v for k, v in base_m.items()})
        row.update({f"candidate_{k}": v for k, v in cand_m.items()})
        row.update(trans)
        row["delta_recall"] = row["candidate_recall"] - row["baseline_recall"]
        row["delta_fpr"] = row["candidate_fpr"] - row["baseline_fpr"]
        row["delta_precision"] = row["candidate_precision"] - row["baseline_precision"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_recall", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--bindingdb-dir", default="../DPI/datasets/BindingDB_AIBind")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline_predictions)
    candidate = pd.read_csv(args.candidate_predictions)
    keep_cols = ["head", "tail", "label", "prob", "pred"]
    baseline = baseline[keep_cols].rename(columns={"prob": "baseline_prob", "pred": "baseline_pred"})
    candidate = candidate[keep_cols].rename(columns={"prob": "candidate_prob", "pred": "candidate_pred"})
    merged = baseline.merge(candidate, on=["head", "tail", "label"], how="inner")

    smiles = {
        strip_prefix(k, "Drug"): v
        for k, v in load_pickle(Path(args.bindingdb_dir) / "feature" / "compound_smiles.pkl").items()
    }
    merged["smiles"] = merged["head"].astype(str).map(smiles)
    merged["motif_bucket"] = merged["smiles"].map(motif_bucket)

    labels = merged["label"].astype(int).values
    base_pred = merged["baseline_pred"].astype(int).values
    cand_prob = merged["candidate_prob"].astype(float).values
    base_metrics = metrics(labels, base_pred)

    pred_pos_threshold = pick_threshold_for_pred_pos(cand_prob, base_metrics["pred_pos"])
    fpr_threshold = pick_threshold_for_fpr(labels, cand_prob, base_metrics["fpr"])

    scenarios = {
        "original_0p5": 0.5,
        "matched_pred_pos": pred_pos_threshold,
        "matched_fpr": fpr_threshold,
    }

    scenario_rows = []
    for name, threshold in scenarios.items():
        scenario_df = add_predictions(merged, threshold)
        candidate_pred = scenario_df["candidate_pred_matched"].astype(int).values
        cand_metrics = metrics(labels, candidate_pred)
        transitions = transition_counts(scenario_df, "candidate_pred_matched")
        scenario_rows.append(
            {
                "scenario": name,
                "threshold": threshold,
                **{f"baseline_{k}": v for k, v in base_metrics.items()},
                **{f"candidate_{k}": v for k, v in cand_metrics.items()},
                **transitions,
                "net_pos_rescue": transitions["rescued_fns"] - transitions["lost_tps"],
                "net_fp_change": transitions["new_fps"] - transitions["fixed_fps"],
            }
        )
        bucket = bucket_summary(scenario_df, "candidate_pred_matched")
        bucket.to_csv(output_dir / f"bucket_summary_{name}.csv", index=False)

    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_summary.to_csv(output_dir / "threshold_scenario_summary.csv", index=False)
    print(scenario_summary.to_string(index=False))

    print("\nMatched predicted-positive bucket summary:")
    cols = [
        "motif_bucket",
        "pos",
        "baseline_recall",
        "candidate_recall",
        "delta_recall",
        "baseline_fpr",
        "candidate_fpr",
        "delta_fpr",
        "baseline_precision",
        "candidate_precision",
        "delta_precision",
        "rescued_fns",
        "lost_tps",
        "fixed_fps",
        "new_fps",
    ]
    print(pd.read_csv(output_dir / "bucket_summary_matched_pred_pos.csv")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
