from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from data.wikitext import get_wikitext

from libprune import prune_wanda, check_sparsity
from libprune import layer_cut
from models import load_from_path, ModelBase

from transformers import HfArgumentParser

def execute_wanda_prune(model, prune_kwargs):
    dataset = prune_kwargs.get('dataset', 'wikitext')
    data = None
    if dataset == 'wikitext':
        data = get_wikitext(train_size=prune_kwargs.get('train_size', 128), test_size=0)
    else: 
        assert False

    use_variant = prune_kwargs.get('use_variant', False) 
    
    # Sparsity type either unstructured-{ratio}, 2:4 or 4:8
    sparsity_type = prune_kwargs['sparsity_type']
    sparsity_ratio = 0.5
    prune_n, prune_m = 0, 0
    if 'unstructured' in sparsity_type:
        sparsity_ratio = float(sparsity_type.split('-')[1])
        sparsity_type = 'unstructured'
    else:
        prune_n, prune_m = map(int, sparsity_type.split(':'))

    prune_wanda(model, data, use_variant, sparsity_ratio, prune_n=prune_n, prune_m=prune_m)
    return model

def execute_layercut(model: ModelBase, prune_kwargs):
    dataset_name = prune_kwargs.get('dataset_name', 'wikitext')
    train_size = prune_kwargs.get('train_size', 250)
    metric_name = prune_kwargs.get('dist_metric', 'angles')
    metric = None 
    if metric_name == 'angles':
        metric = layer_cut.compute_angular_distance 
    elif metric_name == 'distil':
        metric = layer_cut.compute_distil_loss
    else:
        assert False 

    data = None 
    if dataset_name == 'wikitext':
        data = get_wikitext(train_size, 0, True)
    else:
        assert False

    cut_strategy = prune_kwargs.get('cut_strategy', 'simple')
    skip_layers = prune_kwargs.get('skip_layers', None)
    if skip_layers is None:
        sparsity_ratio = prune_kwargs['sparsity_ratio']
        skip_layers = int(model.n_layers * sparsity_ratio)

    if cut_strategy == 'simple':
        model = layer_cut.simple_prune(model, data, skip_layers, dist_metric=metric)
    elif cut_strategy == 'iter':
        model = layer_cut.iter_prune(model, data, skip_layers, dist_metric=metric)

    return model

@dataclass
class PruneConfig:
    prune_method: str 
    model_path: str 
    model_out: str
    prune_kwargs_path: str 

config = HfArgumentParser(PruneConfig).parse_args_into_dataclasses()[0]

model = load_from_path(config.model_path)
prune_kwargs_path = Path(config.prune_kwargs_path) 
prune_kwargs = json.loads(prune_kwargs_path.read_text())
if config.prune_method == 'wanda':
    # TODO: Remove this for inference, proper device management 
    model.for_inference()
    model = execute_wanda_prune(model, prune_kwargs)
    check_sparsity(model, log_modules=False)
elif config.prune_method == 'layer_cut':
    model.for_inference()
    model = execute_layercut(model, prune_kwargs)
else:
    assert False

model.model.save_pretrained(config.model_out)
model.tokenizer.save_pretrained(config.model_out)
