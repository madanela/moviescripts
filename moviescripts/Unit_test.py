



import pytest
import torch
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig

from moviescripts.trainer.trainer import SentenceClassifier,SentenceClassifierEncoded

from pytorch_lightning import Trainer
from typing import List
import os

from moviescripts import get_parameters

import pandas as pd
import numpy as np






initialize(config_path="conf")
global_config = compose("config.yaml",return_hydra_config=True)



def get_config():
    global global_config
    return global_config


def get_model():
    cfg = get_config()
    return SentenceClassifierEncoded(cfg)




def test_config():
    config = get_config()

    assert(config.general.project_name == "moviescripts")
    assert(config.general.experiment_name == "baseline")

    assert(config.loss == {'_target_': 'torch.nn.CrossEntropyLoss', 'ignore_index': '${data.ignore_label}'})



def test_model_shape():
    cfg = get_config()
    model = get_model()

    batch_size = 4
    in_channels = cfg.model.in_channels
    input_tensor = torch.rand((batch_size, in_channels))
    output = model(input_tensor)
    assert output.shape == (batch_size, cfg.model.num_labels) 


def test_model_shape():
    cfg = get_config()
    model = get_model()

    batch_size = 4
    in_channels = cfg.model.in_channels
    input_tensor = torch.rand((batch_size, in_channels))
    output = model(input_tensor)
    assert output.shape == (batch_size, cfg.model.num_labels) 

def test_dataset():

    cfg = get_config()
    model = get_model()

    # Default value
    data_dir = "data/raw/starwars"


    folder_ep4 = os.path.join(data_dir,"SW_EpisodeIV.txt")
    folder_ep5 = os.path.join(data_dir,"SW_EpisodeV.txt")
    folder_ep6 = os.path.join(data_dir,"SW_EpisodeVI.txt")
    assert os.path.exists(folder_ep4), f"Error: {folder_ep4} does not exist."
    assert os.path.exists(folder_ep5), f"Error: {folder_ep5} does not exist."
    assert os.path.exists(folder_ep6), f"Error: {folder_ep6} does not exist."

    df_ep4 = pd.read_csv(folder_ep4, sep =' ', header=0, escapechar='\\')
    df_ep5 = pd.read_csv(folder_ep5, sep =' ', header=0, escapechar='\\')
    df_ep6 = pd.read_csv(folder_ep6, sep =' ', header=0, escapechar='\\')
    Y = pd.concat([df_ep4['character'],df_ep5['character'],df_ep6['character']]).tolist()
    X = pd.concat([df_ep4['dialogue'],df_ep5['dialogue'],df_ep6['dialogue']]).tolist()
    labels = np.unique(Y)
    label_count = [sum(i == np.array(Y)) for i in labels]
    
    min_len_X = min([len(i) for i in X])
    max_len_X = max([len(i) for i in X])
    assert(min_len_X>=3)
    assert(max_len_X<=800)
    min_word_X = min([len(i.split()) for i in X])
    max_word_X = max([len(i.split()) for i in X])
    assert(min_word_X>=1)
    assert(max_word_X<=135)


    df = pd.read_csv(cfg.data.train_dataset.data_dir)

    y = df.y

    assert(len(np.unique(y))==cfg.model.num_labels)

# test_dataset()