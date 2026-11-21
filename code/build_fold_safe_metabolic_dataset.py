import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

import pandas as pd


def strip_prefix(value, prefix):
    return str(value).replace(f"{prefix}:", "")


def sha1_text(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalize_pair_df(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["head"] = df["head"].map(lambda x: strip_prefix(x, "Drug"))
    df["tail"] = df["tail"].map(lambda x: strip_prefix(x, "Protein"))
    df["label"] = df["label"].astype(float).astype(int)
    return df[["head", "tail", "label"]]


def load_feature_txt_from_lines(lines, prefix):
    values = {}
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        values[strip_prefix(parts[0], prefix)] = parts[1]
    return values


def load_metabolic_dataset(path):
    path = Path(path)
    if path.is_dir():
        train = pd.read_csv(path / "blind_start" / "train_set0.csv", index_col=0)
        val = pd.read_csv(path / "blind_start" / "val_set0.csv", index_col=0)
        drug_lines = (path / "feature" / "drug_list.txt").read_text().splitlines()
        protein_lines = (path / "feature" / "protein_list.txt").read_text().splitlines()
    else:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            train_name = next(n for n in names if n.endswith("blind_start/train_set0.csv"))
            val_name = next(n for n in names if n.endswith("blind_start/val_set0.csv"))
            drug_name = next(n for n in names if n.endswith("feature/drug_list.txt"))
            protein_name = next(n for n in names if n.endswith("feature/protein_list.txt"))
            train = pd.read_csv(zf.open(train_name), index_col=0)
            val = pd.read_csv(zf.open(val_name), index_col=0)
            drug_lines = zf.read(drug_name).decode("utf-8", errors="replace").splitlines()
            protein_lines = zf.read(protein_name).decode("utf-8", errors="replace").splitlines()

    return (
        normalize_pair_df(train),
        normalize_pair_df(val),
        load_feature_txt_from_lines(drug_lines, "Drug"),
        load_feature_txt_from_lines(protein_lines, "Protein"),
    )


def load_bindingdb_fold(bindingdb_dir, set_id):
    base = Path(bindingdb_dir) / "blind_start"
    return {
        split: normalize_pair_df(pd.read_csv(base / f"{split}_set{set_id}.csv", index_col=0))
        for split in ["train", "val", "test"]
    }


def load_bindingdb_features(bindingdb_dir):
    import pickle

    with open(Path(bindingdb_dir) / "feature" / "compound_smiles.pkl", "rb") as handle:
        drugs = {strip_prefix(k, "Drug"): v for k, v in pickle.load(handle).items()}
    with open(Path(bindingdb_dir) / "feature" / "protein_aas.pkl", "rb") as handle:
        proteins = {strip_prefix(k, "Protein"): v for k, v in pickle.load(handle).items()}
    return drugs, proteins


def annotate(df, drugs, proteins):
    out = df.copy()
    out["drug_value"] = out["head"].map(drugs)
    out["protein_value"] = out["tail"].map(proteins)
    return out


def filter_df(df, heldout_drugs, heldout_proteins, max_drug_degree):
    keep = df["drug_value"].notna() & df["protein_value"].notna()
    keep &= ~df["drug_value"].isin(heldout_drugs)
    keep &= ~df["protein_value"].isin(heldout_proteins)

    if max_drug_degree and max_drug_degree > 0:
        degrees = df.loc[keep].groupby("head")["tail"].size()
        allowed = set(degrees[degrees <= max_drug_degree].index.astype(str))
        keep &= df["head"].astype(str).isin(allowed)

    return df.loc[keep].copy()


def remap_ids(train, val, drugs, proteins):
    used_drugs = sorted(set(train["head"]) | set(val["head"]), key=lambda x: int(x) if str(x).isdigit() else str(x))
    used_proteins = sorted(set(train["tail"]) | set(val["tail"]), key=lambda x: int(x) if str(x).isdigit() else str(x))
    drug_map = {old: str(i) for i, old in enumerate(used_drugs)}
    protein_map = {old: str(i) for i, old in enumerate(used_proteins)}

    def apply_map(df):
        out = df.copy()
        out["head"] = out["head"].map(drug_map)
        out["tail"] = out["tail"].map(protein_map)
        return out[["head", "tail", "label"]].reset_index(drop=True)

    return apply_map(train), apply_map(val), drug_map, protein_map


def write_feature_list(path, prefix, mapping, values):
    reverse = {new: old for old, new in mapping.items()}
    with open(path, "w", newline="\n") as handle:
        for new_id in sorted(reverse, key=lambda x: int(x)):
            old_id = reverse[new_id]
            handle.write(f"{prefix}:{new_id} {values[old_id]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindingdb-dir", default="../DPI/datasets/BindingDB_AIBind")
    parser.add_argument("--metabolic-path", default="../DPI/datasets/BindingDB_AIBind/metabolic_pathway_phase_2.zip")
    parser.add_argument("--set-id", type=int, default=4)
    parser.add_argument("--output-dataset-dir", default="../DPI/datasets/metabolic_pathway_phase_2_fold4_strict")
    parser.add_argument("--max-drug-degree", type=int, default=0, help="0 disables high-degree drug filtering")
    args = parser.parse_args()

    bdb_drugs, bdb_proteins = load_bindingdb_features(args.bindingdb_dir)
    fold = load_bindingdb_fold(args.bindingdb_dir, args.set_id)
    heldout = pd.concat([fold["val"], fold["test"]], ignore_index=True)
    heldout = annotate(heldout, bdb_drugs, bdb_proteins)
    heldout_drugs = set(heldout["drug_value"].dropna())
    heldout_proteins = set(heldout["protein_value"].dropna())

    train, val, metabolic_drugs, metabolic_proteins = load_metabolic_dataset(args.metabolic_path)
    train = annotate(train, metabolic_drugs, metabolic_proteins)
    val = annotate(val, metabolic_drugs, metabolic_proteins)

    before = {"train": len(train), "val": len(val)}
    train = filter_df(train, heldout_drugs, heldout_proteins, args.max_drug_degree)
    val = filter_df(val, heldout_drugs, heldout_proteins, args.max_drug_degree)
    after = {"train": len(train), "val": len(val)}

    remapped_train, remapped_val, drug_map, protein_map = remap_ids(train, val, metabolic_drugs, metabolic_proteins)

    out_dir = Path(args.output_dataset_dir)
    (out_dir / "blind_start").mkdir(parents=True, exist_ok=True)
    (out_dir / "feature").mkdir(parents=True, exist_ok=True)
    remapped_train.to_csv(out_dir / "blind_start" / "train_set0.csv")
    remapped_val.to_csv(out_dir / "blind_start" / "val_set0.csv")
    write_feature_list(out_dir / "feature" / "drug_list.txt", "Drug", drug_map, metabolic_drugs)
    write_feature_list(out_dir / "feature" / "protein_list.txt", "Protein", protein_map, metabolic_proteins)

    with open(out_dir / "feature" / "id_mapping.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entity_type", "old_id", "new_id", "value_sha1", "value_len", "value_if_drug"])
        writer.writeheader()
        for old_id, new_id in sorted(drug_map.items(), key=lambda x: int(x[1])):
            value = metabolic_drugs[old_id]
            writer.writerow({
                "entity_type": "drug",
                "old_id": old_id,
                "new_id": new_id,
                "value_sha1": sha1_text(value),
                "value_len": len(value),
                "value_if_drug": value,
            })
        for old_id, new_id in sorted(protein_map.items(), key=lambda x: int(x[1])):
            value = metabolic_proteins[old_id]
            writer.writerow({
                "entity_type": "protein",
                "old_id": old_id,
                "new_id": new_id,
                "value_sha1": sha1_text(value),
                "value_len": len(value),
                "value_if_drug": "",
            })

    print(f"Saved fold-safe metabolic dataset: {out_dir}")
    print(f"BindingDB fold: {args.set_id}")
    print(f"Rows before filter: train={before['train']} val={before['val']}")
    print(f"Rows after filter:  train={after['train']} val={after['val']}")
    print(f"Unique remapped drugs={len(drug_map)} proteins={len(protein_map)}")
    if args.max_drug_degree:
        print(f"Applied max_drug_degree={args.max_drug_degree}")


if __name__ == "__main__":
    main()
