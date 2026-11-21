import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def read_split(path):
    df = pd.read_csv(path, index_col=0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.reset_index(drop=True)


def stratified_sample(df, frac, seed):
    parts = []
    for _, group in df.groupby("label", sort=False):
        parts.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def copy_feature_files(source_dir, output_dir):
    feature_out = output_dir / "feature"
    feature_out.mkdir(parents=True, exist_ok=True)
    for name in ["drug_list.txt", "protein_list.txt", "id_mapping.csv"]:
        src = source_dir / "feature" / name
        if src.exists():
            shutil.copy2(src, feature_out / name)


def write_dataset(train, val, output_dir):
    split_out = output_dir / "blind_start"
    split_out.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_out / "train_set0.csv")
    val.to_csv(split_out / "val_set0.csv")


def split_summary(train, val):
    train_drugs, val_drugs = set(train["head"].astype(str)), set(val["head"].astype(str))
    train_prots, val_prots = set(train["tail"].astype(str)), set(val["tail"].astype(str))
    train_pairs = set(zip(train["head"].astype(str), train["tail"].astype(str)))
    val_pairs = set(zip(val["head"].astype(str), val["tail"].astype(str)))
    return {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_pos": int(train["label"].sum()),
        "train_neg": int((train["label"] == 0).sum()),
        "val_pos": int(val["label"].sum()),
        "val_neg": int((val["label"] == 0).sum()),
        "train_unique_drugs": int(train["head"].nunique()),
        "val_unique_drugs": int(val["head"].nunique()),
        "train_unique_proteins": int(train["tail"].nunique()),
        "val_unique_proteins": int(val["tail"].nunique()),
        "train_val_drug_overlap": int(len(train_drugs & val_drugs)),
        "train_val_protein_overlap": int(len(train_prots & val_prots)),
        "train_val_pair_overlap": int(len(train_pairs & val_pairs)),
        "val_unseen_drug_row_frac": float((~val["head"].astype(str).isin(train_drugs)).mean()),
        "val_unseen_protein_row_frac": float((~val["tail"].astype(str).isin(train_prots)).mean()),
    }


def make_random_pair_split(source_dir, output_dir, val_frac, seed):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    train = read_split(source_dir / "blind_start" / "train_set0.csv")
    val = read_split(source_dir / "blind_start" / "val_set0.csv")
    all_rows = pd.concat([train.assign(source_split="train"), val.assign(source_split="val")], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["head", "tail", "label"]).reset_index(drop=True)

    val_new = stratified_sample(all_rows, val_frac, seed)
    val_index = set(val_new.index.tolist())
    train_new = all_rows.loc[~all_rows.index.isin(val_index)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_new = val_new.reset_index(drop=True)

    write_dataset(train_new, val_new, output_dir)
    copy_feature_files(source_dir, output_dir)
    summary = split_summary(train_new, val_new)
    summary.update({"mode": "random_pair", "source_dir": str(source_dir), "output_dir": str(output_dir), "val_frac": val_frac, "seed": seed})
    (output_dir / "diagnostic_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", default="../DPI/datasets/metabolic_pathway_phase_2_fold4_strict")
    parser.add_argument("--output-dataset-dir", default="../DPI/datasets/metabolic_pathway_phase_2_fold4_strict_random_pair_split")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    summary = make_random_pair_split(args.source_dataset_dir, args.output_dataset_dir, args.val_frac, args.seed)
    print(json.dumps(summary, indent=2))
    print("Saved:", os.path.abspath(args.output_dataset_dir))


if __name__ == "__main__":
    main()
