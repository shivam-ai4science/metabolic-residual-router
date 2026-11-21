import argparse
import hashlib
import pickle
import zipfile
from pathlib import Path

import pandas as pd


def strip_prefix(value, prefix):
    value = str(value)
    return value.replace(f"{prefix}:", "")


def sha1_text(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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

    train = normalize_pair_df(train)
    val = normalize_pair_df(val)
    train["metabolic_split"] = "train"
    val["metabolic_split"] = "val"
    pairs = pd.concat([train, val], ignore_index=True)
    drugs = load_feature_txt_from_lines(drug_lines, "Drug")
    proteins = load_feature_txt_from_lines(protein_lines, "Protein")
    return pairs, drugs, proteins


def load_bindingdb_fold(bindingdb_dir, set_id):
    base = Path(bindingdb_dir) / "blind_start"
    dfs = {}
    for split in ["train", "val", "test"]:
        dfs[split] = normalize_pair_df(pd.read_csv(base / f"{split}_set{set_id}.csv", index_col=0))
    return dfs


def annotate_pairs(df, drug_values, protein_values):
    out = df.copy()
    out["drug_value"] = out["head"].map(drug_values)
    out["protein_value"] = out["tail"].map(protein_values)
    out["pair_key"] = list(zip(out["drug_value"], out["protein_value"]))
    out["drug_missing"] = out["drug_value"].isna()
    out["protein_missing"] = out["protein_value"].isna()
    return out


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


def pair_label_map(df):
    result = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.drug_value) or pd.isna(row.protein_value):
            continue
        result.setdefault(row.pair_key, set()).add(int(row.label))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindingdb-dir", default="../DPI/datasets/BindingDB_AIBind")
    parser.add_argument("--metabolic-path", default="../DPI/datasets/BindingDB_AIBind/metabolic_pathway_phase_2.zip")
    parser.add_argument("--output-dir", default="reports/metabolic_transfer_audit")
    parser.add_argument("--set-ids", default="0,1,2,3,4")
    args = parser.parse_args()

    bindingdb_dir = Path(args.bindingdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bdb_drugs = {strip_prefix(k, "Drug"): v for k, v in load_pickle(bindingdb_dir / "feature" / "compound_smiles.pkl").items()}
    bdb_proteins = {strip_prefix(k, "Protein"): v for k, v in load_pickle(bindingdb_dir / "feature" / "protein_aas.pkl").items()}

    metabolic_pairs, metabolic_drugs, metabolic_proteins = load_metabolic_dataset(args.metabolic_path)
    metabolic_pairs = annotate_pairs(metabolic_pairs, metabolic_drugs, metabolic_proteins)
    metabolic_pairs["motif_bucket"] = metabolic_pairs["drug_value"].map(motif_bucket)

    bdb_smiles = set(bdb_drugs.values())
    bdb_sequences = set(bdb_proteins.values())
    metabolic_smiles = set(metabolic_drugs.values())
    metabolic_sequences = set(metabolic_proteins.values())

    summary_rows = []
    overlap_rows = []
    set_ids = [int(x.strip()) for x in args.set_ids.split(",") if x.strip()]

    for set_id in set_ids:
        fold = {
            split: annotate_pairs(df, bdb_drugs, bdb_proteins)
            for split, df in load_bindingdb_fold(bindingdb_dir, set_id).items()
        }
        split_pair_maps = {split: pair_label_map(df) for split, df in fold.items()}

        heldout_drugs = set(fold["val"]["drug_value"].dropna()) | set(fold["test"]["drug_value"].dropna())
        heldout_proteins = set(fold["val"]["protein_value"].dropna()) | set(fold["test"]["protein_value"].dropna())

        heldout_drug_mask = metabolic_pairs["drug_value"].isin(heldout_drugs)
        heldout_protein_mask = metabolic_pairs["protein_value"].isin(heldout_proteins)

        exact_counts = {}
        conflict_counts = {}
        for split, mapping in split_pair_maps.items():
            exact = 0
            conflict = 0
            for row in metabolic_pairs.itertuples(index=False):
                labels = mapping.get(row.pair_key)
                if labels is None:
                    continue
                exact += 1
                same = int(row.label) in labels
                if not same:
                    conflict += 1
                overlap_rows.append({
                    "fold": set_id,
                    "bindingdb_split": split,
                    "metabolic_split": row.metabolic_split,
                    "metabolic_head": row.head,
                    "metabolic_tail": row.tail,
                    "metabolic_label": int(row.label),
                    "bindingdb_labels": ";".join(map(str, sorted(labels))),
                    "same_label": same,
                    "drug_sha1": sha1_text(str(row.drug_value)),
                    "protein_sha1": sha1_text(str(row.protein_value)),
                    "motif_bucket": row.motif_bucket,
                })
            exact_counts[split] = exact
            conflict_counts[split] = conflict

        summary_rows.append({
            "fold": set_id,
            "metabolic_rows": len(metabolic_pairs),
            "metabolic_pos": int(metabolic_pairs["label"].sum()),
            "metabolic_neg": int((metabolic_pairs["label"] == 0).sum()),
            "unique_metabolic_drugs": metabolic_pairs["head"].nunique(),
            "unique_metabolic_proteins": metabolic_pairs["tail"].nunique(),
            "exact_drug_smiles_overlap_unique": len(metabolic_smiles & bdb_smiles),
            "exact_protein_sequence_overlap_unique": len(metabolic_sequences & bdb_sequences),
            "heldout_val_test_drug_rows": int(heldout_drug_mask.sum()),
            "heldout_val_test_protein_rows": int(heldout_protein_mask.sum()),
            "heldout_val_test_either_rows": int((heldout_drug_mask | heldout_protein_mask).sum()),
            "exact_train_pair_rows": exact_counts["train"],
            "exact_val_pair_rows": exact_counts["val"],
            "exact_test_pair_rows": exact_counts["test"],
            "conflict_train_pair_rows": conflict_counts["train"],
            "conflict_val_pair_rows": conflict_counts["val"],
            "conflict_test_pair_rows": conflict_counts["test"],
        })

    drug_degree = (
        metabolic_pairs.groupby(["head", "drug_value", "motif_bucket"], dropna=False)
        .agg(rows=("label", "size"), positives=("label", "sum"), unique_proteins=("tail", "nunique"))
        .reset_index()
    )
    drug_degree["negatives"] = drug_degree["rows"] - drug_degree["positives"]
    drug_degree["in_bindingdb_exact_smiles"] = drug_degree["drug_value"].isin(bdb_smiles)
    drug_degree = drug_degree.sort_values(["rows", "positives"], ascending=False)

    motif_summary = (
        metabolic_pairs.groupby("motif_bucket")
        .agg(rows=("label", "size"), positives=("label", "sum"), unique_drugs=("head", "nunique"), unique_proteins=("tail", "nunique"))
        .reset_index()
    )
    motif_summary["negatives"] = motif_summary["rows"] - motif_summary["positives"]
    motif_summary = motif_summary.sort_values("rows", ascending=False)

    pd.DataFrame(summary_rows).to_csv(output_dir / "fold_leakage_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_dir / "exact_pair_overlaps.csv", index=False)
    drug_degree.to_csv(output_dir / "metabolic_drug_degree.csv", index=False)
    motif_summary.to_csv(output_dir / "metabolic_motif_summary.csv", index=False)

    print(f"Saved audit reports to {output_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nTop metabolic drug degrees:")
    print(drug_degree.head(20)[["head", "rows", "positives", "negatives", "unique_proteins", "motif_bucket", "in_bindingdb_exact_smiles"]].to_string(index=False))


if __name__ == "__main__":
    main()
