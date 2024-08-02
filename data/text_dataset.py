from dataclasses import dataclass
from datasets import Dataset

@dataclass 
class TextDataset:
    train_data: Dataset
    test_data: Dataset