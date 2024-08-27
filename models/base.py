import abc
from enum import Enum
import json
from pathlib import Path
from typing import Dict, Optional

import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Qwen2ForCausalLM,
    LlamaForCausalLM,
    GemmaForCausalLM,
    OPTForCausalLM,
)

class ModelType(Enum): 
    LLAMA = "llama"
    QWEN2 = "qwen2"
    GEMMA = "gemma"
    OPT   = "opt"

def to_semi_sparse(model: nn.Module):
    '''
    Mark all linear modules as sparse_semi_structured for torch
    '''
    from torch.sparse import to_sparse_semi_structured
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        try:
            module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
        except KeyboardInterrupt:
            raise 
        except Exception as e:
            print(f'Could not make {name} semi-structured due to {e}.')
        
    return model

def to_dense(model: nn.Module):
    '''
    Make all linear modules dense (reverse to_semi_sparse)
    '''
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        module.weight = nn.Parameter(module.weight.to_dense())
        
    return model


PrunnableModel = Qwen2ForCausalLM | LlamaForCausalLM | GemmaForCausalLM | OPTForCausalLM

class PruneMethod(Enum):
    WANDA = "wanda"
    LAYERCUT = "layer_cut"

class ModelBase:
    def __init__(self, model_path: str, model_type: ModelType, 
                 prune_method: Optional[PruneMethod]=None, prune_kwargs: Optional[Dict[str, str]]=None, 
                 **kwargs):
        self._model_path = model_path
        self._model_type = model_type 
        self.prune_method = prune_method 
        self.prune_kwargs = prune_kwargs

        self._load()
        assert self.model_type is not None, 'Unknown model'

    @property
    def model_path(self):
        return self._model_path

    @property
    def model_type(self):
        return self._model_type
     
    def dump(self, root: Path, with_tokenizer=True):
        self.model.save_pretrained(root)
        if with_tokenizer:
            self.tokenizer.save_pretrained(root)
        if self.prune_method is not None:
            prune_cfg_path = root.joinpath('prunekit_config.json')
            prune_cfg_path.write_text(json.dumps({
                'prune_method': self.prune_method.value,
                'prune_kwargs': self.prune_kwargs 
            }))

    def _load(self):
        # TODO: Peft models, assisted models...
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path) 
    
    def for_inference(self):
        self.model.eval()
        # TODO: Do better for accelerate
        self._model = self.model.cuda()
        if self.prune_method == PruneMethod.WANDA and self.prune_kwargs.get('sparsity_type') in ('2:4', '4:8'):
            print('Detected wanda prune, converting to semi-sparse')
            self._model = to_semi_sparse(self.model)

    def for_training(self):
        # Add the pad token?
        assert False, 'Not implemented'
    
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer
    
    @property
    def model(self) -> PrunnableModel:
        return self._model

    @property
    @abc.abstractmethod
    def decoder_layers_path(self) -> str:
        ...
    
    def _model_getattr_rec(self, path: str):
        sub_attrs = path.split('.')
        x = self.model
        for attr_name in sub_attrs:
            x = getattr(x, attr_name)    
        
        return x 

    def _model_setattr_rec(self, path: str, new_value):
        sub_attrs = path.split('.')
        path_minus_one = '.'.join(sub_attrs[:-1])
        final_attr_name = sub_attrs[-1]
        x_minus_one = self._model_getattr_rec(path_minus_one)
        setattr(x_minus_one, final_attr_name, new_value)

    def get_decoder_layers(self) -> nn.ModuleList:
        return self._model_getattr_rec(self.decoder_layers_path)

    def set_decoder_layers(self, x: nn.ModuleList):
        self._model_setattr_rec(self.decoder_layers_path, x)
        self.model.config.num_hidden_layers = len(x)

    @property
    def n_layers(self):
        return len(self.get_decoder_layers())
