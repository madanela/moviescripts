from collections import MutableMapping

import torch
from loguru import logger


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_baseline_model(cfg, model):
    # if it is Minkoski weights
    cfg.model.in_channels = 512

    model = model(cfg)
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    model.model.load_state_dict(state_dict)
    return cfg, model



def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key, None) is None:
            print("eh")
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if state_dict[key].shape != correct_dict[key].shape:
            logger.warning(
                f"incorrect shape {key}:{state_dict[key].shape} vs {correct_dict[key].shape}"
            )
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if key in correct_dict.keys():
            new_state_dict.update({key: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model