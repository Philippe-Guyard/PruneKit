# Implementation of Unreasonable Inneffectiveness of the deeper layers 

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TextDataset, TokenizedDataset
from .utils import with_module_io_hooks
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

        # (n_layers, bsz, seq_len, hidden_size)
        layer_embeddings = torch.stack([
            get_layer_embeddings(layer_idx)
            for layer_idx in range(model.n_layers)
        ])
        # (n_layers, bsz, seq_len)
        cos_sim = F.cosine_similarity(layer_embeddings[:-skip_layers], layer_embeddings[skip_layers:], dim=-1) 
        angular_dist = torch.acos(cos_sim) / torch.pi
        distances.append(angular_dist.view(model.n_layers - skip_layers, -1).cpu())
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

    distances = torch.cat(distances, dim=1)
    return distances.mean(dim=1)
        
def simple_prune(model: ModelBase, data: TextDataset, skip_layers: int=1):
    distances = compute_angular_distance(model, data, skip_layers)
    for idx, dist in enumerate(distances):
        # Make layers 1-indexed
        print(f'Dist(In(Layer {idx + 1}), Out(Layer {idx + skip_layers})) = {dist.item():.2f}')

    start_layer = distances.argmin()
    print(f'Removing layers in range [{start_layer}, {start_layer + skip_layers - 1}]')
    new_layers = [l for l in model.get_decoder_layers()]
    new_layers = nn.ModuleList(new_layers[:start_layer] + new_layers[start_layer + skip_layers:])
    model.set_decoder_layers(new_layers)
    return model

def iter_prune(model: ModelBase, data: TextDataset, skip_layers:int=1):
    '''
    Instead of removing n consecutive layers, remove 1 layer at a time n times
    '''
    for _ in range(skip_layers):
        model = simple_prune(model, data, skip_layers=1)
    
    return model