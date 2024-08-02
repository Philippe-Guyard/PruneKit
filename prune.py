from argparse import ArgumentParser
import json
from typing import Literal

from data.wikitext import get_wikitext

from libprune import prune_wanda, check_sparsity
from models import load_from_path 

def execute_wanda_prune(model, prune_kwargs):
    prune_kwargs = prune_config['prune_kwargs']
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

parser = ArgumentParser()
parser.add_argument('--config')

args = parser.parse_args()
config_path = args.config

prune_config = None
with open(config_path, 'r') as config_file:
    prune_config = json.load(config_file)

model = load_from_path(prune_config['model_path'])
prune_kwargs = prune_config.get('prune_kwargs', dict())
if prune_config['prune_method'] == 'wanda':
    # TODO: Remove this for inference, proper device management 
    model.for_inference()
    model = execute_wanda_prune(model, prune_kwargs)
    check_sparsity(model, log_modules=False)
else:
    assert False

model.model.save_pretrained(prune_config['model_out'])
model.tokenizer.save_pretrained(prune_config['model_out'])
