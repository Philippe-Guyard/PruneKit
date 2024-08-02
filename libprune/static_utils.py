from typing import Dict
from models import ModelBase

import torch.nn as nn

def get_linear_children(module: nn.Module) -> Dict[str, nn.Linear]:
    """
    Find all nn.Linear layers in a module.
    Args:
        module (nn.Module): PyTorch module.
    Returns:
        dict: Dictionary of nn.Linear layers within the module.
    """
    linear_layers = {}

    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            linear_layers[name] = submodule

    return linear_layers

def check_sparsity(model: ModelBase, log_modules=True, log_layers=True, log_model=True):
    model_zeros, model_params = 0, 0
    for idx, decoder_layer in enumerate(model.get_decoder_layers()):
        linears = get_linear_children(decoder_layer)
        layer_zeros, layer_params = 0, 0
        for name, layer in linears.items():
            W = layer.weight.data
            zeros = (W==0).sum().item()
            params = W.numel()
            if log_modules:
                print(f'Layer {idx}, module {name} has sparsity {zeros / params:.2f}')

            layer_zeros += zeros
            layer_params += params
        
        if log_layers:
            print(f'Layer {idx} has sparsity {layer_zeros / layer_params:.2f}')
    
        model_zeros += layer_zeros
        model_params += layer_params

    if log_model:
        print(f'Model has sparsity {model_zeros / model_params:.2f}')