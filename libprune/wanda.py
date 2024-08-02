
from typing import Dict
from models import ModelBase
from data import TextDataset

from .static_utils import get_linear_children

import torch
import torch.nn as nn

class WrappedGPT:
    """
    Borrowed from https://github.com/locuslab/wanda/blob/main/lib/layerwrapper.py
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # Borrowed from https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

# TODO: This and the prune wanda can be rewritten but I don't have time 
def prepare_calibration_input(model: ModelBase, data: TextDataset, device):
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    cache = {'hidden_states': [], 'attention_mask': [], "position_ids": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, **kwargs):
            cache['hidden_states'].append(hidden_states)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs.get('position_ids', None))
            raise ValueError

    orig_layers = model.get_decoder_layers()
    model.set_decoder_layers(torch.nn.ModuleList([Catcher(orig_layers[0])]))
    for x in data.train_data:
        max_len = min(model.model.config.max_position_embeddings, 8192) 
        tokens = model.tokenizer(x['text'], max_length=max_len, truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.to(device)
        try:
            model.model(input_ids)
        except ValueError:
            pass 

    model.set_decoder_layers(orig_layers)
    model.model.config.use_cache = use_cache

    return cache['hidden_states'], cache['attention_mask'], cache['position_ids'] 

def prune_wanda(model: ModelBase, data: TextDataset, use_variant: bool, sparsity_ratio: float, prune_n=0, prune_m=0):
    # Mostly borrowed from https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py
    use_cache = model.model.config.use_cache 
    model.model.config.use_cache = False 

    n_samples = len(data.train_data)

    device = 'cuda' # model.hf_device_map.get('model.layers.0', 'cuda') # TODO: Can be better
    inps, attention_mask, position_ids = prepare_calibration_input(model, data, device)
    outs = [torch.zeros_like(inps[j]) for j in range(len(inps))] 

    for idx, layer in enumerate(model.get_decoder_layers()):
        prunnable_modules = get_linear_children(layer)

        # TODO: 
        # For multi-gpu inference, might need to change device
        # handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        # if f"model.layers.{idx}" in model.hf_device_map:
        #     dev = model.hf_device_map[f"model.layers.{idx}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in prunnable_modules:
            wrapped_layers[name] = WrappedGPT(prunnable_modules[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # ===========================
        # Single forward pass to do all the add_batch
        handles = []
        for name in wrapped_layers:
            handles.append(prunnable_modules[name].register_forward_hook(add_batch(name)))

        # TODO: Can we make this a single layer call?
        for j in range(n_samples):
            with torch.no_grad():
                kwargs = {'attention_mask': attention_mask[j]}
                if position_ids[j] is not None:
                    kwargs['position_ids'] = position_ids[j]
                outs[j] = layer(inps[j], **kwargs)[0]

        for h in handles:
            h.remove()
        
        # ====================

        for name in prunnable_modules:
            print(f"pruning layer {idx} name {name}")
            W_metric = torch.abs(prunnable_modules[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            prunnable_modules[name].weight.data[W_mask] = 0  ## set weights to zero 

        # Actually recompute outs...
        for j in range(n_samples):
            with torch.no_grad():
                kwargs = {'attention_mask': attention_mask[j]}
                if position_ids[j] is not None:
                    kwargs['position_ids'] = position_ids[j]
                outs[j] = layer(inps[j], **kwargs)[0]
        inps, outs = outs, inps

    model.model.config.use_cache = use_cache 
    torch.cuda.empty_cache()