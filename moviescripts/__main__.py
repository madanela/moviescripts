from sys import platform

import os
import logging

from dotenv import load_dotenv
from uuid import uuid4


import hydra
import torch
from git import Repo
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from hashlib import md5
from moviescripts import __version__

from moviescripts.utils.utils import (
    flatten_dict,load_baseline_model,load_checkpoint_with_missing_or_exsessive_keys
)
from moviescripts.trainer.trainer import SentenceClassifier,SentenceClassifierEncoded
from sentence_transformers import SentenceTransformer

from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    # making shure reproducable
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting accelerator
    if cfg.general.get("accelerator", None) is None: 
        if platform == "darwin":
            cfg.general.accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            cfg.general.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    # getting basic configuration
    if cfg.general.get("devices", None) is None:
        cfg.general.devices = 'auto'
    print("devices!! :",cfg.general.devices)
    loggers = []

    cfg.general.experiment_id = str(Repo("./").commit())[:8]
    params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    unique_id = "_" + str(uuid4())[:4]
    cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id
    print("version: ",cfg.general.version)
    for log in cfg.logging:
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )
    print("logging passed!")
    model = SentenceClassifierEncoded(cfg)
    if cfg.general.checkpoint is not None:
        if cfg.general.checkpoint[-3:] == "pth":
            # loading model weights, if it has .pth in the end, it will work with it
            # as if it work with original Minkowski weights
            cfg, model = load_baseline_model(cfg, SentenceClassifierEncoded)
        else:
            cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
 
    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def predict(model_path: str, inputs: str) -> str:
    # load the saved model
    model = SentenceClassifierEncoded.load_from_checkpoint(model_path)
    model.freeze()
    encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')

    model_input = encoder_model.encode(inputs)

    inputs = model(model_input)


    # process the input
    # ...

    # make predictions
    # ...

    # return the predicted output
    return predicted_output


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg : DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model ,loggers = get_parameters(cfg)

    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))
    
    runner = Trainer(
        devices=cfg.general.devices,
        accelerator=cfg.general.accelerator,
        logger=loggers,
        default_root_dir=str(cfg.general.save_dir),
        **cfg.trainer,
    )

    runner.fit(model)



@hydra.main(config_path="conf", config_name="config.yaml")
def test(cfg : DictConfig): 
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model ,loggers = get_parameters(cfg)

    runner = Trainer(
        devices=cfg.general.devices,
        accelerator=cfg.general.accelerator,
        logger=loggers,
        default_root_dir=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.test(model)
train()


