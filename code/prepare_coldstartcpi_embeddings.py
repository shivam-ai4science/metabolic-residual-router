from pathlib import Path
import argparse
import pickle

import numpy as np
import torch


REQUIRED_FEATURES = [
    "compound_Mol2Vec300.pkl",
    "compound_Atom2Vec300.pkl",
    "aas_ProtTransBertBFD1024.pkl",
]


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def as_tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def pack_global_and_local(global_feature, local_features):
    global_tensor = as_tensor(global_feature).reshape(1, -1)
    local_tensor = as_tensor(local_features)
    if local_tensor.ndim == 1:
        local_tensor = local_tensor.reshape(1, -1)
    if global_tensor.size(1) != local_tensor.size(1):
        raise ValueError(f"global dim {global_tensor.size(1)} != local dim {local_tensor.size(1)}")
    return torch.cat([global_tensor, local_tensor], dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent.parent / "DPI" / "datasets" / "BindingDB_AIBind"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    feature_dir = dataset_dir / "feature"

    missing = [name for name in REQUIRED_FEATURES if not (feature_dir / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing official ColdstartCPI feature files in {feature_dir}: {names}")

    drug_global = load_pickle(feature_dir / "compound_Mol2Vec300.pkl")
    drug_local = load_pickle(feature_dir / "compound_Atom2Vec300.pkl")
    protein_features = load_pickle(feature_dir / "aas_ProtTransBertBFD1024.pkl")

    common_drugs = sorted(set(drug_global) & set(drug_local))
    skipped_drugs = []
    ligand_embeddings = {}
    for drug_id in common_drugs:
        try:
            ligand_embeddings[drug_id] = pack_global_and_local(drug_global[drug_id], drug_local[drug_id])
        except (TypeError, ValueError, RuntimeError) as exc:
            skipped_drugs.append((drug_id, str(exc)))

    skipped_proteins = []
    protein_embeddings = {}
    for protein_id, features in protein_features.items():
        try:
            protein_embeddings[protein_id] = pack_global_and_local(features[0], features[1])
        except (TypeError, ValueError, RuntimeError) as exc:
            skipped_proteins.append((protein_id, str(exc)))

    drug_dir = dataset_dir / "embeddings" / "drugs"
    protein_dir = dataset_dir / "embeddings" / "proteins_prottrans_bfd"
    drug_dir.mkdir(parents=True, exist_ok=True)
    protein_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ligand_embeddings, drug_dir / "ligand_embeddings_coldstartcpi.pt")
    torch.save(protein_embeddings, protein_dir / "protein_embeddings_coldstartcpi.pt")

    first_ligand = next(iter(ligand_embeddings.values()))
    first_protein = next(iter(protein_embeddings.values()))
    print(f"Saved {len(ligand_embeddings)} ColdstartCPI ligand tensors; first shape {tuple(first_ligand.shape)}")
    print(f"Saved {len(protein_embeddings)} ColdstartCPI protein tensors; first shape {tuple(first_protein.shape)}")
    if skipped_drugs:
        print(f"Skipped {len(skipped_drugs)} ligand feature entries:")
        for drug_id, reason in skipped_drugs[:20]:
            print(f"  {drug_id}: {reason}")
    if skipped_proteins:
        print(f"Skipped {len(skipped_proteins)} protein feature entries:")
        for protein_id, reason in skipped_proteins[:20]:
            print(f"  {protein_id}: {reason}")


if __name__ == "__main__":
    main()
