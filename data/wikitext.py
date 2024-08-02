from .text_dataset import TextDataset

from typing import Optional

from datasets import load_dataset

def get_wikitext(train_size: Optional[int]=None, test_size: Optional[int]=None, filter_nonempty=True):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    train_data = wikitext['train']
    test_data = wikitext['test']

    if filter_nonempty:
        not_empty = lambda x: len(x['text']) > 0
        train_data = train_data.filter(not_empty)
        test_data  = test_data.filter(not_empty)
    
    if train_size is not None:
        train_data = train_data.select(range(train_size))
    if test_size is not None:
        test_data = test_data.select(range(test_size))

    return TextDataset(train_data, test_data)
