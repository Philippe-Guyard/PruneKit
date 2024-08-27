import json
from pathlib import Path
from typing import Type

from .opt import OptModel
from .gated import Qwen2Model, GatedModel
from .base import ModelBase, ModelType, PruneMethod

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
    prune_method, prune_kwargs = None, None
    if model_is_local:
        root = Path(model_path)
        model_basename = json.loads(root.joinpath('config.json').read_text())['model_type'] 
        prune_cfg_path = root.joinpath('prunekit_config.json')
        if prune_cfg_path.exists():
            prune_cfg = json.loads(prune_cfg_path.read_text())
            prune_method = PruneMethod(prune_cfg['prune_method'])
            prune_kwargs = prune_cfg['prune_kwargs']

    model_type = infer_model_type(model_basename)
    model_cls = MODEL_TYPE_TO_CLS[model_type]
    return model_cls(model_path, model_type, prune_method=prune_method, prune_kwargs=prune_kwargs)

def load_lmeval_obj(model_path: str, batch_size):
    # TODO: Assisted models
    return LMEvalHFBase(
        pretrained=model_path,
        batch_size=batch_size
    )
