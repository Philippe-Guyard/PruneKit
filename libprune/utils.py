from typing import Dict

import torch
from models import ModelBase

import torch.nn as nn
import torch.nn.functional as F

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

def kldiv_loss(student_logits, teacher_logits, per_batch=False, temperature=1.0):
    """
    Compute the KL-divergence between student and teacher logits.

    Parameters:
    - student_logits: Logits from the student model (tensor of shape [seq_len, vocab_size])
    - teacher_logits: Logits from the teacher model (tensor of shape [seq_len, vocab_size])
    - temperature: Temperature for softening the probability distributions
    - per_batch: If True, return a loss value for every token 
    """
    # Soften the probabilities with temperature
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Compute the KL divergence loss
    reduction = 'none' if per_batch else 'batchmean'
    kldiv_loss = F.kl_div(student_probs, teacher_probs, reduction=reduction)
    if per_batch:
        kldiv_loss = kldiv_loss.sum(dim=1)
    
    kldiv_loss *= (temperature ** 2)
    return kldiv_loss

def distillation_loss(student_logits, teacher_logits, labels, per_batch=False, temperature=1.0, alpha=0.5):
    """
    Compute the distillation loss between student and teacher,
    which is defined as alpha * KL divergence + (1 - alpha) * Cross Entropy loss with true labels

    Parameters:
    - student_logits: Logits from the student model (tensor of shape [batch_size, vocab_size])
    - teacher_logits: Logits from the teacher model (tensor of shape [batch_size, vocab_size])
    - labels: Ground truth labels (tensor of shape [batch_size])
    - temperature: Temperature for softening the probability distributions
    - alpha: Weighting factor for balancing the KL divergence and cross-entropy loss

    Returns:
    - Loss value (scalar tensor)
    """
    kldiv = kldiv_loss(student_logits, teacher_logits, per_batch=per_batch, temperature=temperature)
    
    # Compute the cross-entropy loss with true labels
    cross_entropy_loss = F.cross_entropy(student_logits, labels, reduction='none' if per_batch else 'mean')
    if per_batch:
        cross_entropy_loss = cross_entropy_loss.mean(dim=0)
    
    # Combine the two losses
    combined_loss = alpha * kldiv + (1 - alpha) * cross_entropy_loss
    
    return combined_loss