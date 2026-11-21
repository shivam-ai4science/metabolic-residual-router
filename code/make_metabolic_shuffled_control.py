import argparse
import shutil
from pathlib import Path

import pandas as pd


def shuffle_tails_within_label(df, seed):
    out = df.copy()
    rng = __import__("random").Random(seed)
    for label in sorted(out["label"].unique()):
        idx = out.index[out["label"] == label].tolist()
        tails = out.loc[idx, "tail"].tolist()
        rng.shuffle(tails)
        out.loc[idx, "tail"] = tails
    exact_same = ((out["head"] == df["head"]) & (out["tail"] == df["tail"])).sum()
    if exact_same:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-dir", required=True)
    parser.add_argument("--output-dataset-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    src = Path(args.input_dataset_dir)
    dst = Path(args.output_dataset_dir)
    (dst / "blind_start").mkdir(parents=True, exist_ok=True)
    (dst / "feature").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        df = pd.read_csv(src / "blind_start" / f"{split}_set0.csv", index_col=0)
        shuffled = shuffle_tails_within_label(df, args.seed + (0 if split == "train" else 1))
        shuffled.to_csv(dst / "blind_start" / f"{split}_set0.csv")
        unchanged = ((df["head"].astype(str) == shuffled["head"].astype(str)) & (df["tail"].astype(str) == shuffled["tail"].astype(str))).sum()
        print(f"{split}: rows={len(df)} unchanged_exact_positions={unchanged}")

    for name in ["drug_list.txt", "protein_list.txt", "id_mapping.csv"]:
        src_file = src / "feature" / name
        if src_file.exists():
            shutil.copy2(src_file, dst / "feature" / name)

    print(f"Saved shuffled control dataset: {dst}")
    print("This preserves label counts and marginal head/tail multisets within each label, but breaks true pair identity.")


if __name__ == "__main__":
    main()
