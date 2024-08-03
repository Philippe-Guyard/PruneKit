import itertools

from models import ModelBase

from dataclasses import dataclass
from datasets import Dataset

@dataclass 
class TextDataset:
    train_data: Dataset
    test_data: Dataset

class TokenizedDataset:
    DEFAULT_MAX_LEN = 4096

    def __init__(self, data: Dataset, model: ModelBase, batch_size: int = 1, max_length=None):
        self.data = data
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length or TokenizedDataset.DEFAULT_MAX_LEN

        self._done = True
        self._data_iter = None 

    def __len__(self):
        # divide and round up
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._done = False
        self._data_iter = iter(self.data)
        return self 

    def __next__(self):
        if self._done:
            raise StopIteration()

        texts = []
        while len(texts) < self.batch_size:
            try:
                texts.append(next(self._data_iter)['text'])
            except StopIteration:
                self._done = True
                break
        
        if len(texts) == 0:
            raise StopIteration

        max_length = min(self.max_length, self.model.model.config.max_position_embeddings)
        tokens = self.model.tokenizer(
            texts, return_tensors='pt',
            truncation=True, padding='longest', max_length=max_length
        )
        return tokens 

    

    