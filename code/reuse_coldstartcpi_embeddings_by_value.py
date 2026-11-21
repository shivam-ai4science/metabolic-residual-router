import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import torch


ENTITY_SPECS = {
    "drugs": {
        "prefix": "Drug",
        "list_name": "drug_list.txt",
        "pickle_name": "compound_smiles.pkl",
        "embedding_relpath": Path("embeddings/drugs/ligand_embeddings_coldstartcpi.pt"),
    },
    "proteins": {
        "prefix": "Protein",
        "list_name": "protein_list.txt",
        "pickle_name": "protein_aas.pkl",
        "embedding_relpath": Path(
            "embeddings/proteins_prottrans_bfd/protein_embeddings_coldstartcpi.pt"
        ),
    },
}


def canonical_id(value, prefix):
    value = str(value)
    return value if value.startswith(f"{prefix}:") else f"{prefix}:{value}"


def load_values(dataset_dir, spec):
    feature_dir = dataset_dir / "feature"
    list_path = feature_dir / spec["list_name"]
    if list_path.exists():
        values = {}
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split(maxsplit=1)
            if len(parts) == 2:
                values[canonical_id(parts[0], spec["prefix"])] = parts[1]
        return values

    pickle_path = feature_dir / spec["pickle_name"]
    if pickle_path.exists():
        with open(pickle_path, "rb") as handle:
            raw_values = pickle.load(handle)
        return {
            canonical_id(key, spec["prefix"]): str(value)
            for key, value in raw_values.items()
        }

    raise FileNotFoundError(
        f"Neither {list_path} nor {pickle_path} exists for {dataset_dir}"
    )


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_shape_is_valid(tensor, entity_type):
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        return False
    expected_dim = 300 if entity_type == "drugs" else 1024
    return tensor.shape[0] >= 1 and tensor.shape[1] == expected_dim


def recover_entity_embeddings(entity_type, target_dir, source_dirs):
    spec = ENTITY_SPECS[entity_type]
    target_values = load_values(target_dir, spec)
    recovered = {}
    source_counts = {}
    source_consistency = {}

    for source_dir in source_dirs:
        missing_ids = [key for key in target_values if key not in recovered]
        if not missing_ids:
            break

        embedding_path = source_dir / spec["embedding_relpath"]
        if not embedding_path.exists():
            print(f"Skipping source without packed {entity_type}: {source_dir}")
            continue

        source_values = load_values(source_dir, spec)
        source_embeddings_raw = safe_torch_load(embedding_path)
        source_embeddings = {
            canonical_id(key, spec["prefix"]): tensor
            for key, tensor in source_embeddings_raw.items()
        }
        # A sequence/SMILES may occur under multiple IDs. Prefer the first ID
        # that actually has a packed tensor instead of letting a missing alias
        # hide another valid embedding for the same exact entity value.
        value_to_source_id = {}
        for source_id, value in source_values.items():
            if source_id in source_embeddings:
                value_to_source_id.setdefault(value, source_id)

        recovered_by_value = {
            target_values[target_id]: tensor
            for target_id, tensor in recovered.items()
        }
        common_values = [
            value
            for value in recovered_by_value
            if value in value_to_source_id
            and value_to_source_id[value] in source_embeddings
        ]
        checked = 0
        max_abs_diff = 0.0
        mismatches = []
        if common_values:
            stride = max(1, len(common_values) // 64)
            for value in common_values[::stride][:64]:
                left = recovered_by_value[value]
                right = source_embeddings[value_to_source_id[value]]
                checked += 1
                if left.shape != right.shape:
                    mismatches.append((value[:80], tuple(left.shape), tuple(right.shape)))
                    continue
                difference = float((left - right).abs().max().item())
                max_abs_diff = max(max_abs_diff, difference)
                if not torch.allclose(left, right, rtol=1e-5, atol=1e-5):
                    mismatches.append((value[:80], difference))
            if mismatches:
                raise ValueError(
                    f"Cross-source {entity_type} embedding mismatch for {source_dir}: "
                    f"{mismatches[:10]}"
                )
        source_consistency[str(source_dir.resolve())] = {
            "shared_values_checked": checked,
            "max_abs_difference": max_abs_diff,
        }

        added = 0
        invalid = []
        for target_id in missing_ids:
            source_id = value_to_source_id.get(target_values[target_id])
            if source_id is None or source_id not in source_embeddings:
                continue
            tensor = source_embeddings[source_id]
            if not tensor_shape_is_valid(tensor, entity_type):
                invalid.append((target_id, source_id, tuple(tensor.shape)))
                continue
            recovered[target_id] = tensor
            added += 1

        if invalid:
            raise ValueError(
                f"Invalid {entity_type} tensors from {source_dir}: {invalid[:10]}"
            )
        source_counts[str(source_dir.resolve())] = added
        print(
            f"{entity_type}: recovered {added} from {source_dir}; "
            f"total {len(recovered)}/{len(target_values)}"
        )

        del source_embeddings
        del source_embeddings_raw

    missing = [key for key in target_values if key not in recovered]
    if missing:
        preview = [(key, target_values[key][:80]) for key in missing[:20]]
        raise RuntimeError(
            f"Could not recover {len(missing)} of {len(target_values)} {entity_type}. "
            f"First missing values: {preview}"
        )

    output_path = target_dir / spec["embedding_relpath"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_path.parent).free
    largest_source = max(
        (source_dir / spec["embedding_relpath"]).stat().st_size
        for source_dir in source_dirs
        if (source_dir / spec["embedding_relpath"]).exists()
    )
    required_bytes = largest_source + 5 * 1024**3
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk for atomic {entity_type} save: "
            f"free={free_bytes / 1024**3:.1f} GiB, "
            f"required={required_bytes / 1024**3:.1f} GiB"
        )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(recovered, temporary_path)
    os.replace(temporary_path, output_path)
    print(
        f"Saved {len(recovered)} {entity_type} tensors to {output_path} "
        f"({output_path.stat().st_size / 1024**3:.2f} GiB)"
    )
    return {
        "target_count": len(target_values),
        "source_counts": source_counts,
        "source_consistency": source_consistency,
        "output_path": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-key packed ColdstartCPI tensors by exact SMILES/protein-sequence "
            "identity. No labels or model predictions are read."
        )
    )
    parser.add_argument("--target-dataset-dir", required=True)
    parser.add_argument(
        "--source-dataset-dir",
        action="append",
        required=True,
        help="Repeat in preferred lookup order.",
    )
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dataset_dir).resolve()
    source_dirs = [Path(path).resolve() for path in args.source_dataset_dir]

    if not target_dir.exists():
        raise FileNotFoundError(target_dir)
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)

    summary = {
        "method": "exact_entity_value_rekey_v1",
        "target_dataset_dir": str(target_dir),
        "source_dataset_dirs": [str(path) for path in source_dirs],
        "uses_labels": False,
        "uses_predictions": False,
    }
    for entity_type in ["drugs", "proteins"]:
        summary[entity_type] = recover_entity_embeddings(
            entity_type, target_dir, source_dirs
        )

    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else target_dir / "embedding_reuse_manifest.json"
    )
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    print("EXACT-VALUE EMBEDDING REUSE OK")


if __name__ == "__main__":
    main()
