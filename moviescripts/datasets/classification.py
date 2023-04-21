import random

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from moviescripts.datasets.augmentation import AddNoise,str_to_float_list

import pandas as pd
import numpy as np

class TextClassificationEncodedDataset(Dataset):
    def __init__(self,data_dir,mode: Optional[str] = "train"):
        df = pd.read_csv(data_dir)

        self.X = df.X
        self.y = df.y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        text = self.X[idx]
        label = self.y[idx]
        return torch.tensor(str_to_float_list(text)), torch.tensor(label)
    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self.y
class TextClassificationDataset(Dataset):
    def __init__(self, data_dir, 
                 tokenizer,augment: Optional[str] = None,
                 mode: Optional[str] = "train"):

        df = pd.read_csv(data_dir)

        self.X = df.X
        self.y = df.y
        self.tokenizer = tokenizer
        self.augment = augment

        # if working only on classes for validation - discard others


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = str(self.X[idx])
        if self.augment:
            text = self.augment(text, p = .8)

        # text = AddNoise(str(self.X[idx]), p = .8)
        label = self.y[idx]

        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, torch.tensor(label)
    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self.y