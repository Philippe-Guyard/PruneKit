import abc
from enum import Enum
from typing import Optional

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

from lm_eval.models.huggingface import HFLM as LMEvalHFBase

class ModelType(Enum): 
    LLAMA = "llama"
    QWEN2 = "qwen2"
    GEMMA = "gemma"
    OPT   = "opt"

PrunnableModel = Qwen2ForCausalLM | LlamaForCausalLM | GemmaForCausalLM | OPTForCausalLM

class ModelBase:
    def __init__(self, model_path: str, model_type: ModelType, **kwargs):
        self._model_path = model_path
        self._model_type = model_type 

        self._load()
        assert self.model_type is not None, 'Unknown model'

    @property
    def model_path(self):
        return self._model_path

    @property
    def model_type(self):
        return self._model_type
    
    def _load(self):
        # TODO: Peft models, assisted models...
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path) 
    
    def for_inference(self):
        self.model.eval()
        # TODO: Do better for accelerate
        self._model = self.model.cuda()

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
