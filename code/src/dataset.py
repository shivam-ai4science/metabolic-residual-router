import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor


def trim_tensor(tensor, max_tokens):
    if max_tokens is None or tensor.size(0) <= max_tokens:
        return tensor
    return tensor[:max_tokens]


def padding_mask_from_lengths(tensors):
    lengths = torch.tensor([tensor.size(0) for tensor in tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    positions = torch.arange(max_len).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def pad_global_local_tensor(tensor, local_tokens):
    if tensor.ndim != 2 or tensor.size(0) < 1:
        raise ValueError("Expected a [global + local_tokens, feature_dim] tensor")

    output = tensor.new_zeros((local_tokens + 1, tensor.size(1)))
    output[0] = tensor[0]
    copied_local_tokens = min(max(tensor.size(0) - 1, 0), local_tokens)
    if copied_local_tokens:
        output[1 : copied_local_tokens + 1] = tensor[1 : copied_local_tokens + 1]

    padding_mask = torch.ones(local_tokens + 1, dtype=torch.bool)
    padding_mask[: copied_local_tokens + 1] = False
    return output, padding_mask


class OfficialFixedPaddingCollate:
    """Reconstruct the official 100-atom/1000-residue collate semantics."""

    def __init__(self, prot_local_tokens=1000, lig_local_tokens=100):
        self.prot_local_tokens = int(prot_local_tokens)
        self.lig_local_tokens = int(lig_local_tokens)

    def __call__(self, batch):
        prot_list, lig_list, prot_masks, lig_masks = [], [], [], []
        labels_list, weights_list, metas = [], [], []
        for prot, lig, label, weight, meta in batch:
            prot_padded, prot_mask = pad_global_local_tensor(prot, self.prot_local_tokens)
            lig_padded, lig_mask = pad_global_local_tensor(lig, self.lig_local_tokens)
            prot_list.append(prot_padded)
            lig_list.append(lig_padded)
            prot_masks.append(prot_mask)
            lig_masks.append(lig_mask)
            labels_list.append(label)
            weights_list.append(weight)
            metas.append(meta)

        return (
            torch.stack(prot_list),
            torch.stack(lig_list),
            torch.stack(prot_masks),
            torch.stack(lig_masks),
            torch.stack(labels_list),
            torch.stack(weights_list),
            metas,
        )


def resolve_protein_tensor_path(prot_dir, protein_id):
    candidates = [
        os.path.join(prot_dir, f"{protein_id}.pt"),
        os.path.join(prot_dir, f"{protein_id.replace(':', '_')}.pt"),
        os.path.join(prot_dir, f"{protein_id.split(':')[-1]}.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def sample_weight_from_row(row):
    if "sample_weight" in row:
        return row["sample_weight"]
    if "weight" in row:
        return row["weight"]
    return 1.0


def metadata_from_row(row, idx):
    meta = {"row_index": int(idx), "head": str(row["head"]), "tail": str(row["tail"])}
    for key in [
        "oof_base_prob",
        "oof_base_logit",
        "oof_teacher_count",
        "oof_teacher_std",
        "teacher_fold",
    ]:
        if key in row:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                meta[key] = value if key != "teacher_fold" else int(value)
    return meta


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


class DPIDatasetStatic(Dataset):
    def __init__(self, df, protein_embedding, ligand_embeddings, max_prot_tokens=None, max_lig_tokens=None):
        self.df = df
        self.protein_embedding = protein_embedding
        self.ligand_embeddings = ligand_embeddings
        self.max_prot_tokens = max_prot_tokens
        self.max_lig_tokens = max_lig_tokens
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = f"Drug:{row['head']}"
        protein = f"Protein:{str(row['tail'])}"
        label = torch.tensor(row['label'], dtype=torch.float32)
        weight = torch.tensor(sample_weight_from_row(row), dtype=torch.float32)
        
        prot_tensor = trim_tensor(self.protein_embedding[protein], self.max_prot_tokens)
        lig_tensor = trim_tensor(self.ligand_embeddings[smiles], self.max_lig_tokens)
        
        meta = metadata_from_row(row, idx)
        return prot_tensor, lig_tensor, label, weight, meta

def static_collate_fn(batch):
    prot_list, lig_list, labels_list, weights_list, metas = [], [], [], [], []
    for prot, lig, label, weight, meta in batch:
        prot_list.append(prot)
        lig_list.append(lig)
        labels_list.append(label)
        weights_list.append(weight)
        metas.append(meta)
        
    prot_padded = pad_sequence(prot_list, batch_first=True, padding_value=0.0)
    lig_padded = pad_sequence(lig_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels_list)
    weights = torch.stack(weights_list)
    prot_padding_mask = padding_mask_from_lengths(prot_list)
    lig_padding_mask = padding_mask_from_lengths(lig_list)
    
    return prot_padded, lig_padded, prot_padding_mask, lig_padding_mask, labels, weights, metas

def get_cached_protein(path):
    return safe_torch_load(path)

class ThreadSafeDPIDataset(Dataset):
    def __init__(self, df, prot_dir, ligand_embeddings):
        self.df = df
        self.prot_dir = prot_dir
        self.ligand_embeddings = ligand_embeddings
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = f"Drug:{row['head']}"
        protein = f"Protein:{str(row['tail'])}"
        label = torch.tensor(row['label'], dtype=torch.float32)
        weight = torch.tensor(sample_weight_from_row(row), dtype=torch.float32)
        
        prot_tensor = resolve_protein_tensor_path(self.prot_dir, protein)
        lig_tensor = self.ligand_embeddings[smiles] 
        
        meta = metadata_from_row(row, idx)
        return prot_tensor, lig_tensor, label, weight, meta

class ThreadedCollate:
    def __init__(self, num_threads=8, max_prot_tokens=None, max_lig_tokens=None):
        self.num_threads = num_threads
        self.max_prot_tokens = max_prot_tokens
        self.max_lig_tokens = max_lig_tokens

    def _load_single_pair(self, item):
        uid_path, lig, label, weight, meta = item
        prot = trim_tensor(get_cached_protein(uid_path), self.max_prot_tokens)
        lig = trim_tensor(lig, self.max_lig_tokens)
        return prot, lig, label, weight, meta

    def __call__(self, batch):
        prot_list, lig_list, labels_list, weights_list, metas = [], [], [], [], []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(self._load_single_pair, batch))
            
        for prot, lig, label, weight, meta in results:
            prot_list.append(prot)
            lig_list.append(lig)
            labels_list.append(label)
            weights_list.append(weight)
            metas.append(meta)
            
        prot_padded = pad_sequence(prot_list, batch_first=True, padding_value=0.0)
        lig_padded = pad_sequence(lig_list, batch_first=True, padding_value=0.0)
        labels = torch.stack(labels_list)
        weights = torch.stack(weights_list)
        prot_padding_mask = padding_mask_from_lengths(prot_list)
        lig_padding_mask = padding_mask_from_lengths(lig_list)
        
        return prot_padded, lig_padded, prot_padding_mask, lig_padding_mask, labels, weights, metas
    
class DPIDatasetMultiProcessing(Dataset):
    def __init__(self, df, prot_dir, ligand_embeddings, max_prot_tokens=None, max_lig_tokens=None):
        self.df = df
        self.prot_dir = prot_dir
        self.ligand_embeddings = ligand_embeddings
        self.max_prot_tokens = max_prot_tokens
        self.max_lig_tokens = max_lig_tokens
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = f"Drug:{row['head']}"
        protein = f"Protein:{str(row['tail'])}"
        label = torch.tensor(row['label'], dtype=torch.float32)
        weight = torch.tensor(sample_weight_from_row(row), dtype=torch.float32)
        
        # Load directly here; the DataLoader workers will execute this in parallel
        prot_tensor_path = resolve_protein_tensor_path(self.prot_dir, protein)
        prot_tensor = trim_tensor(safe_torch_load(prot_tensor_path), self.max_prot_tokens)
        lig_tensor = trim_tensor(self.ligand_embeddings[smiles], self.max_lig_tokens)
        meta = metadata_from_row(row, idx)
        
        return prot_tensor, lig_tensor, label, weight, meta

def custom_collate_multiprocessing(batch):
    prot_list, lig_list, labels_list, weights_list, metas = [], [], [], [], []
    
    for prot, lig, label, weight, meta in batch:
        prot_list.append(prot)
        lig_list.append(lig)
        labels_list.append(label)
        weights_list.append(weight)
        metas.append(meta)
        
    # --- STANDARD PADDING ---
    prot_padded = pad_sequence(prot_list, batch_first=True, padding_value=0.0)
    lig_padded = pad_sequence(lig_list, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels_list)
    weights = torch.stack(weights_list)
    
    prot_padding_mask = padding_mask_from_lengths(prot_list)
    lig_padding_mask = padding_mask_from_lengths(lig_list)
    
    return prot_padded, lig_padded, prot_padding_mask, lig_padding_mask, labels, weights, metas


def get_dataloaders(config_dict, train_df, val_df, test_df=None):
    """Factory function to build dataloaders based on config."""
    
    data_config = config_dict['data']
    dataset_path = os.path.join(data_config['base_path'], data_config['dataset_name'])
    if not os.path.exists(dataset_path):
        dataset_path = data_config['base_path']
    batch_size_train = data_config['batch_size_train']
    batch_size_val = data_config['batch_size_val']
    num_workers = data_config.get('num_workers', 0)
    max_prot_tokens = data_config.get("max_prot_tokens")
    max_lig_tokens = data_config.get("max_lig_tokens")
    read_on_fly = data_config.get('read_on_fly', False)
    default_pin_memory = True if read_on_fly else False
    worker_kwargs = {
        "num_workers": num_workers,
        "pin_memory": data_config.get("pin_memory", default_pin_memory),
    }
    if num_workers > 0:
        worker_kwargs["prefetch_factor"] = 1

    if data_config.get("padding_mode", "dynamic") == "official_fixed":
        collate_fn = OfficialFixedPaddingCollate(
            prot_local_tokens=data_config.get("official_prot_local_tokens", 1000),
            lig_local_tokens=data_config.get("official_lig_local_tokens", 100),
        )
    else:
        collate_fn = None
    
    embedding_name = data_config.get("embeddings", "")
    if embedding_name != "":
        ligand_embeddings = safe_torch_load(f"{dataset_path}/embeddings/drugs/ligand_embeddings_{embedding_name}.pt")
    else:
        ligand_embeddings = safe_torch_load(f"{dataset_path}/embeddings/drugs/ligand_embeddings.pt")


    if read_on_fly:
        prot_dir = f"{dataset_path}/embeddings/proteins_prottrans_bfd/"

        train_dataset = DPIDatasetMultiProcessing(train_df, prot_dir, ligand_embeddings, max_prot_tokens, max_lig_tokens)
        val_dataset = DPIDatasetMultiProcessing(val_df, prot_dir, ligand_embeddings, max_prot_tokens, max_lig_tokens)
        test_dataset = DPIDatasetMultiProcessing(test_df, prot_dir, ligand_embeddings, max_prot_tokens, max_lig_tokens) if test_df is not None else None

        active_collate_fn = collate_fn or custom_collate_multiprocessing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=active_collate_fn, **worker_kwargs)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=active_collate_fn, **worker_kwargs)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=active_collate_fn, **worker_kwargs) if test_dataset is not None else None

    else:
        protein_embedding_path = f"{dataset_path}/embeddings/proteins_prottrans_bfd/protein_embeddings.pt"
        if embedding_name != "":
            named_path = f"{dataset_path}/embeddings/proteins_prottrans_bfd/protein_embeddings_{embedding_name}.pt"
            if os.path.exists(named_path):
                protein_embedding_path = named_path
        protein_embeddings = safe_torch_load(protein_embedding_path)
        train_dataset = DPIDatasetStatic(train_df, protein_embeddings, ligand_embeddings, max_prot_tokens, max_lig_tokens)
        val_dataset = DPIDatasetStatic(val_df, protein_embeddings, ligand_embeddings, max_prot_tokens, max_lig_tokens)
        test_dataset = DPIDatasetStatic(test_df, protein_embeddings, ligand_embeddings, max_prot_tokens, max_lig_tokens) if test_df is not None else None

        active_collate_fn = collate_fn or static_collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=active_collate_fn, **worker_kwargs)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=active_collate_fn, **worker_kwargs)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=active_collate_fn, **worker_kwargs) if test_dataset is not None else None

    if test_df is not None:
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader
