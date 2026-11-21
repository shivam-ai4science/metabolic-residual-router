import argparse
import pickle
from pathlib import Path

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


def confusion_metrics(df, pred_col):
    label = df["label"].astype(int)
    pred = df[pred_col].astype(int)
    tp = int(((label == 1) & (pred == 1)).sum())
    fn = int(((label == 1) & (pred == 0)).sum())
    fp = int(((label == 0) & (pred == 1)).sum())
    tn = int(((label == 0) & (pred == 0)).sum())
    return {
        "rows": len(df),
        "pos": int((label == 1).sum()),
        "neg": int((label == 0).sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


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
    merged["transition"] = (
        merged["label"].astype(str)
        + ":"
        + merged["baseline_pred"].astype(str)
        + "->"
        + merged["candidate_pred"].astype(str)
    )

    bucket_rows = []
    for bucket, group in merged.groupby("motif_bucket", dropna=False):
        base = confusion_metrics(group, "baseline_pred")
        cand = confusion_metrics(group, "candidate_pred")
        row = {"motif_bucket": bucket}
        for key, value in base.items():
            row[f"baseline_{key}"] = value
        for key, value in cand.items():
            row[f"candidate_{key}"] = value
        row["delta_recall"] = row["candidate_recall"] - row["baseline_recall"]
        row["delta_fpr"] = row["candidate_fpr"] - row["baseline_fpr"]
        row["delta_precision"] = row["candidate_precision"] - row["baseline_precision"]
        bucket_rows.append(row)

    transitions = (
        merged.groupby(["motif_bucket", "transition"])
        .size()
        .reset_index(name="rows")
        .sort_values(["motif_bucket", "transition"])
    )

    rescued_fns = merged[(merged["label"] == 1) & (merged["baseline_pred"] == 0) & (merged["candidate_pred"] == 1)]
    lost_tps = merged[(merged["label"] == 1) & (merged["baseline_pred"] == 1) & (merged["candidate_pred"] == 0)]
    fixed_fps = merged[(merged["label"] == 0) & (merged["baseline_pred"] == 1) & (merged["candidate_pred"] == 0)]
    new_fps = merged[(merged["label"] == 0) & (merged["baseline_pred"] == 0) & (merged["candidate_pred"] == 1)]

    pd.DataFrame(bucket_rows).sort_values("motif_bucket").to_csv(output_dir / "subgroup_metric_shifts.csv", index=False)
    transitions.to_csv(output_dir / "prediction_transitions_by_bucket.csv", index=False)
    rescued_fns.to_csv(output_dir / "rescued_fns.csv", index=False)
    lost_tps.to_csv(output_dir / "lost_tps.csv", index=False)
    fixed_fps.to_csv(output_dir / "fixed_fps.csv", index=False)
    new_fps.to_csv(output_dir / "new_fps.csv", index=False)

    print(f"Compared {len(merged)} shared prediction rows")
    print(f"Saved subgroup reports to {output_dir}")
    print(pd.DataFrame(bucket_rows).sort_values("motif_bucket").to_string(index=False))


if __name__ == "__main__":
    main()
