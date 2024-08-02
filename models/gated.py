from .base import ModelBase

# Works for Llama, Gemma
class GatedModel(ModelBase):
    @property
    def decoder_layers_path(self):
        return 'model.layers'

class Qwen2Model(GatedModel):
    def for_inference(self):
        super().for_inference()
        # Otherwise we get annoying warnings for qwen2 somehow...
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id