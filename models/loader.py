import json
from pathlib import Path
from typing import Type

from .opt import OptModel
from .gated import Qwen2Model, GatedModel
from .base import ModelBase, ModelType

from lm_eval.models.huggingface import HFLM as LMEvalHFBase

MODEL_TYPE_TO_CLS = {
    ModelType.LLAMA: GatedModel,
    ModelType.GEMMA: GatedModel,
    ModelType.QWEN2: Qwen2Model,
    ModelType.OPT  : OptModel,
}

def infer_model_type(base_name: str) -> ModelType:
    for x in ModelType:
        if x.value in base_name.lower():
            return x

    assert False, 'Could not infer model type' 

def load_from_path(model_path: str) -> ModelBase:
    model_is_local = Path(model_path).exists()
    model_basename = model_path
    if model_is_local:
        with open(Path(model_path).joinpath('config.json'), 'r') as cfg_file:
            model_basename = json.load(cfg_file)['model_type'] 

    model_type = infer_model_type(model_basename)
    model_cls = MODEL_TYPE_TO_CLS[model_type]
    return model_cls(model_path, model_type)

def load_lmeval_obj(model_path: str, batch_size):
    # TODO: Assisted models
    return LMEvalHFBase(
        pretrained=model_path,
        batch_size=batch_size
    )
