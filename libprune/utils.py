from typing import Dict

import torch
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

class ModuleIOCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = [dict()]
        self.sample_idx = 0
        self.token_idx = 0

    @property
    def cur_sample(self):
        return self.data[self.sample_idx]
    
    def save_embedding(self, module_key: str, embedding: torch.Tensor):
        if module_key not in self.cur_sample:
            self.cur_sample[module_key] = []

        self.cur_sample[module_key].append(embedding)

    def next_token(self):
        self.token_idx += 1

    def next_sample(self):
        self.data.append(dict())
        self.sample_idx += 1

def with_module_io_hooks(model: ModelBase, collect_modules=None):
    collect_modules = collect_modules or {'layer'}
    def save_data(module_key: str, cache: ModuleIOCache, 
                  save_hidden_states=False, save_output=False, is_last_module=False):
        def save_data_hook(layer: nn.Module, args, kwargs, output):
            hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
            is_first_token = cache.token_idx == 0 
            save_key = module_key + '_first' if is_first_token else module_key
            emb = None 
            if save_hidden_states:
                emb = hidden_states
            if save_output: 
                emb = output[0] if isinstance(output, tuple) else output

            cache.save_embedding(save_key, emb)
            if is_last_module:
                cache.next_token()

        return save_data_hook

    cache = ModuleIOCache()
    for idx, layer in enumerate(model.get_decoder_layers()):
        layer: nn.Module
        if 'layer' in collect_modules:
            hook = save_data(f'layer{idx}', cache, save_hidden_states=True)
            layer.register_forward_hook(hook, with_kwargs=True)
            if idx == model.n_layers - 1:
                last_hook = save_data(f'layer{idx+1}', cache, save_output=True, is_last_module=True)
                layer.register_forward_hook(last_hook, with_kwargs=True)
    
    return model, cache