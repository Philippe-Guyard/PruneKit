from .base import ModelBase

class OptModel(ModelBase):
    @property
    def decoder_layers_path(self):
        return 'model.decoder.layers'