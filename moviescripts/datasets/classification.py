import random

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from moviescripts.datasets.augmentation import AddNoise

import pandas as pd



class TextClassificationDataset(Dataset):
    def __init__(self, data_dir, tokenizer,augment: Optional[str] = None,mode: Optional[str] = "train",):
        print("heeey")
        df = pd.read_csv(data_dir)
        self.X = df.X
        self.y = df.y
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.augment:
            text = self.augment(str(self.X[idx]), p = .8)
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
