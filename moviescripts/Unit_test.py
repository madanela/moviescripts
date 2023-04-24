



import pytest
import torch
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
from moviescripts.trainer.trainer import SentenceClassifier,SentenceClassifierEncoded

from pytorch_lightning import Trainer
from typing import List
import os

from moviescripts import get_parameters

@pytest.fixture
def model():
    return SentenceClassifierEncoded()


@pytest.fixture
def test_data(cfg):
    # load test dataset from config file
    dataset = your_data_loading_function(cfg.test.dataset.path)
    return dataset


@pytest.fixture
def loss_fn(cfg):
    # load loss function from config file
    loss_fn = hydra.utils.instantiate(cfg.test.loss)
    return loss_fn


@pytest.fixture
def metric(cfg):
    # load evaluation metric from config file
    metric = hydra.utils.instantiate(cfg.test.metric)
    return metric


def test_model_shape(cfg, model, test_data):
    batch_size = cfg.test.batch_size
    num_tokens = cfg.test.num_tokens
    input_tensor = torch.rand((batch_size, num_tokens))
    output = model(input_tensor)
    assert output.shape == (batch_size, 2)  # assuming binary classification


def test_model_output_range(cfg, model, test_data):
    batch_size = cfg.test.batch_size
    num_tokens = cfg.test.num_tokens
    input_tensor = torch.rand((batch_size, num_tokens))
    output = model(input_tensor)
    assert (output >= 0).all() and (output <= 1).all()  # assuming sigmoid activation


def test_model_loss(cfg, model, test_data, loss_fn):
    batch_size = cfg.test.batch_size
    num_tokens = cfg.test.num_tokens
    input_tensor = torch.rand((batch_size, num_tokens))
    target = torch.randint(0, 2, (batch_size,)).float()
    output = model(input_tensor)
    loss = loss_fn(output.view(-1), target)
    assert loss >= 0


def test_model_metric(cfg, model, test_data, metric):
    batch_size = cfg.test.batch_size
    num_tokens = cfg.test.num_tokens
    input_tensor = torch.rand((batch_size, num_tokens))
    target = torch.randint(0, 2, (batch_size,)).float()
    output = model(input_tensor)
    metric_value = metric(output.view(-1), target)
    assert metric_value >= 0




# @pytest.fixture
# def model():
#     return SentenceClassifierEncoded()

# @pytest.fixture
# def test_data(cfg):
#     # load test dataset from config file
#     dataset = your_data_loading_function(cfg.test.dataset.path)
#     return dataset

# @pytest.fixture
# def loss_fn(cfg):
#     # load loss function from config file
#     loss_fn = hydra.utils.instantiate(cfg.test.loss)
#     return loss_fn

# @pytest.fixture
# def metric(cfg):
#     # load evaluation metric from config file
#     metric = hydra.utils.instantiate(cfg.test.metric)
#     return metric

# def test_model_shape(model, test_data):
#     # your test code here

# def test_model_output_range(model, test_data):
#     # your test code here

# def test_model_loss(model, test_data, loss_fn):
#     # your test code here

# def test_model_metric(model, test_data, metric):
#     # your test code here


@pytest.fixture
def model(cfg):
    model = SentenceClassifierEncoded(cfg.model)
    return model

@pytest.fixture
def test_data(cfg):
    # load test dataset from config file
    dataset = your_data_loading_function(cfg.test.dataset.path)
    return dataset

@pytest.fixture
def loss_fn(cfg):
    # load loss function from config file
    loss_fn = hydra.utils.instantiate(cfg.test.loss)
    return loss_fn

@pytest.fixture
def metric(cfg):
    # load evaluation metric from config file
    metric = hydra.utils.instantiate(cfg.test.metric)
    return metric

def test_model_shape(model, test_data):
    # your test code here
    pass
def test_model_output_range(model, test_data):
    # your test code here
    pass
def test_model_loss(model, test_data, loss_fn):
    # your test code here
    pass
def test_model_metric(model, test_data, metric):
    # your test code here
    pass
@hydra.main(config_path="conf", config_name="config.yaml")
def test(cfg: DictConfig):
    # Run all test functions
    pytest.main(["-v"])

