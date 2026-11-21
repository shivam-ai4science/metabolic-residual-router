import os
import torch
import importlib
import torch.nn as nn

def get_loss(config_dict):
    """Dynamically loads loss from torch.nn or local src/losses/ files."""
    loss_name = config_dict['loss']['name']
    loss_params = dict(config_dict['loss'].get('params', {}))
    
    # 1. Try to load from standard PyTorch losses
    if hasattr(nn, loss_name):
        loss_class = getattr(nn, loss_name)
        # Handle pos_weight if it's passed as a float from config
        if 'pos_weight' in loss_params and isinstance(loss_params['pos_weight'], (float, int)):
            loss_params['pos_weight'] = torch.tensor([loss_params['pos_weight']])
        return loss_class(**loss_params)

    # 2. Try to auto-discover in the src/losses/ directory
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = f"src.losses.{filename[:-3]}"
            module = importlib.import_module(module_name)
            
            if hasattr(module, loss_name):
                loss_class = getattr(module, loss_name)
                return loss_class(**loss_params)

    raise ValueError(f"Loss '{loss_name}' not found in torch.nn or src/losses/")
