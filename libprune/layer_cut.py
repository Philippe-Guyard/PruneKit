# Implementation of Unreasonable Inneffectiveness of the deeper layers 

from collections import defaultdict
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TextDataset, TokenizedDataset
from .utils import distillation_loss, kldiv_loss, with_module_io_hooks
from models import load_from_path, ModelBase

from tqdm import tqdm

@torch.no_grad()
def compute_angular_distance(model: ModelBase, data: TextDataset, skip_layers: int=1) -> torch.FloatTensor:
    '''
    Given a number skip_layers of layer to skip, for each layer compute the average angular distance between 
    layer_idx and layer_idx + skip_layers, return these distances  
    '''
    train_data = TokenizedDataset(data.train_data, model, batch_size=1)
    model, module_cache = with_module_io_hooks(model)
    
    def get_layer_embeddings(layer_idx):
        emb = module_cache.cur_sample[f'layer{layer_idx}_first'][0]
        cached = module_cache.cur_sample.get(f'layer{layer_idx}', []) 
        if len(cached) > 0:
            cached = torch.cat(cached, dim=1)
            emb = torch.cat((emb, cached), dim=1)
        
        return emb

    distances = []
    for tokens in tqdm(train_data):
        # TODO: Proper devices...
        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        module_cache.reset()
        model.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=model.tokenizer.eos_token_id,
            use_cache=True,
        )

        # (n_layers + 1, bsz, seq_len, hidden_size)
        layer_embeddings = torch.stack([
            get_layer_embeddings(layer_idx)
            # +1 since we have input(0) and output(model.n_layers)
            for layer_idx in range(model.n_layers + 1)
        ])
        # (n_layers - skip_layers + 1, bsz, seq_len)
        cos_sim = F.cosine_similarity(layer_embeddings[:-skip_layers], layer_embeddings[skip_layers:], dim=-1) 
        angular_dist = torch.acos(cos_sim) / torch.pi
        # (n_layers - skip_layers + 1, bsz * seq_len)
        new_shape = (model.n_layers - skip_layers + 1, -1)
        distances.append(angular_dist.reshape(new_shape).cpu())
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

    distances = torch.cat(distances, dim=1)
    return distances.mean(dim=1)
        
def _compute_prune_loss_base(model: ModelBase, data: TextDataset, skip_layers:int=1, loss='kldiv'):
    '''
    Given a number skip_layers of layer to skip, for each layer compute the 
    average loss value between:
    1) The original model
    2) A model where layers between between layer_idx and layer_idx + skip_layers are removed   
    Loss is either defined as KL divergence if loss='kldiv', otherwise it is distillation loss
    '''
    # TODO: This requires model.n_layers forward passes = model.n_layers ** 2 layer forward passes
    # It can be done faster (in model.n_layers ** 2 / 2 layer forward passes) by not recomputing 
    # earlier layer embeddings.
    # Example: 
    # Slow approach  
    # layer_idx = 0: forward through layers 1, 2, ..., n - 1 
    # layer_idx = 1: forward through layers 0, 2, ..., n - 1
    # layer_idx = 2: forward through layers 0, 1, 3, ..., n - 1
    # ...
    # layer_idx = n - 1: forward through layers 0, 1, ..., n - 2
    # Faster approach
    # layer_idx = 0: as before 
    # layer_idx = 1: as before 
    # layer_idx = 2: just use layer-0-embedding from previous run and feed it into layer 2
    # layer_idx = n: just use layers [0..n - 2] embeddings from previous run and feed it into lm_head
    train_data = TokenizedDataset(data.train_data, model, batch_size=1)
    orig_layers = model.get_decoder_layers()

    def get_logits(input_ids):
        return model.model(
            input_ids.cuda(), 
            use_cache=False, 
            past_key_values=None
        ).logits.squeeze(dim=0)

    losses = torch.zeros((model.n_layers, len(train_data)))
    for data_idx, tokens in enumerate(tqdm(train_data)): 
        labels = tokens.input_ids[:, 1:].squeeze(dim=0)
        input_ids = tokens.input_ids[:, :-1]

        teacher_logits = get_logits(input_ids) 
        # We can remove any layer such that layer_idx+skip_layers <= model.n_layers
        for layer_idx in range(model.n_layers - skip_layers + 1):
            new_layers = orig_layers[:layer_idx] + orig_layers[layer_idx + skip_layers:]
            model.set_decoder_layers(new_layers)
            student_logits = get_logits(input_ids)
            loss = None
            if loss == 'kldiv':
                loss = kldiv_loss(student_logits, teacher_logits) 
            elif loss == 'distillation':
                loss = distillation_loss(student_logits, teacher_logits, labels)
            else:
                assert False
            losses[layer_idx, data_idx] = loss
        
        model.set_decoder_layers(orig_layers)
            
    return losses.mean(dim=1) 


@torch.no_grad()
def compute_kldiv_loss(model: ModelBase, data: TextDataset, skip_layers: int=1) -> torch.FloatTensor:
    return _compute_prune_loss_base(model, data, skip_layers, loss='kldiv')

@torch.no_grad()
def compute_distil_loss(model: ModelBase, data: TextDataset, skip_layers: int=1) -> torch.FloatTensor:
    return _compute_prune_loss_base(model, data, skip_layers, 'distillation')

DistMetric = Callable[[ModelBase, TextDataset, int], torch.FloatTensor]
def simple_prune(model: ModelBase, data: TextDataset, 
                 skip_layers: int=1, dist_metric: DistMetric=compute_angular_distance):
    distances = dist_metric(model, data, skip_layers)
    for idx, dist in enumerate(distances):
        # Make layers 1-indexed
        print(f'Dist(In(Layer {idx + 1}), Out(Layer {idx + skip_layers})) = {dist.item():.2f}')

    start_layer = distances.argmin()
    # Again, layers 1-indexed
    print(f'Removing layers in range [{start_layer + 1}, {start_layer + skip_layers}]')
    new_layers = [l for l in model.get_decoder_layers()]
    new_layers = nn.ModuleList(new_layers[:start_layer] + new_layers[start_layer + skip_layers:])
    model.set_decoder_layers(new_layers)
    return model

def iter_prune(model: ModelBase, data: TextDataset, skip_layers:int=1, dist_metric: DistMetric=compute_angular_distance):
    '''
    Instead of removing n consecutive layers, remove 1 layer at a time n times
    '''
    for _ in range(skip_layers):
        model = simple_prune(model, data, skip_layers=1, dist_metric=dist_metric)
    
    return model