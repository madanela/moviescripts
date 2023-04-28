import logging
import os,sys
import pickle
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
from dotenv import load_dotenv
from git import Repo
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from pytorch_lightning import Trainer, seed_everything
from sentence_transformers import SentenceTransformer

from moviescripts.trainer.trainer import SentenceClassifier, SentenceClassifierEncoded
from moviescripts.utils.utils import flatten_dict, load_baseline_model, load_checkpoint_with_missing_or_exsessive_keys
from moviescripts import __version__ as _version

load_dotenv()

_logger = logging.getLogger(__name__)

sys.path.append('.')

# Load the configuration file
initialize(config_path="conf", job_name="test_app")
_cfg = compose(config_name="config")

# Load the saved model
_model = SentenceClassifierEncoded(_cfg)
if _cfg.general.checkpoint is not None:
    if _cfg.general.checkpoint.endswith(".pth"):
        # loading model weights, if it has .pth in the end, it will work with it
        # as if it work with original Minkowski weights
        print("Loading baseline model...")
        _cfg, _model = load_baseline_model(_cfg, SentenceClassifierEncoded)
    else:
        print("Loading checkpoint with missing or excessive keys...")
        _cfg, _model = load_checkpoint_with_missing_or_exsessive_keys(_cfg, _model)

_model.freeze()

_encoder_model = SentenceTransformer('bert-base-nli-mean-tokens')

print(OmegaConf.to_yaml(_cfg))


def load_dict(data_path):
    with open(data_path, 'rb') as handle:
        dict = pickle.load(handle)
        return dict


def validate_text(text: str) -> str:
    if len(text) < 3 or len(text) > 800:
        return "invalid"
    x = text.split()
    if len(x) < 1 or len(x) > 135:
        return "invalid"
    return "valid"


def predict_class(*, input_data: Union[pd.DataFrame, Dict]) -> Dict:
    inputs = input_data['message']

    errors = validate_text(inputs)
    results = {"predictions": None, "version": _version, "errors": errors}

    if errors == "valid":
        model_input = _encoder_model.encode(inputs)

        out_dir = Path("../data/processed/starwars")

        char2ind = load_dict(out_dir / "char2ind.pickle")
        ind2char = load_dict(out_dir / "ind2char.pickle")


        out = _model(torch.tensor(model_input).unsqueeze(0))

        predictions = ind2char[torch.argmax(out[0]).item()]


        _logger.info(
            f"Making predictions with model version: {_version}. Predictions: {predictions}"
        )
        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results
