
from contextlib import nullcontext

import torch
from torchsummary import summary
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl

from moviescripts.models.metrics import Accuracy

from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification,BertTokenizer
from transformers import BertForPreTraining

import hydra

from tqdm import tqdm




class SentenceClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()


        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad

        # loss
        self.ignore_label = config.data.ignore_label
        self.criterion = hydra.utils.instantiate(config.loss)
        # metrics
        print("current config.metrics")
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.Accuracy = Accuracy()
        # misc
        self.labels_info = dict()

        self.criterion = hydra.utils.instantiate(config.loss)
        # self.bert_freezed =  BertForPreTraining.from_pretrained('distilbert-base-uncased') #BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification")
        
        # model
        self.model = hydra.utils.instantiate(config.model)

        self.labels_info = dict()

    def forward(self, input_ids, attention_mask):

        x = self.model(input_ids,attention_mask)

        return x
    def training_step(self, batch, batch_idx):

        input_ids = batch[0]
        attention_mask = batch[1]
        target = batch[2]


        input_ids.to(self.device)
        attention_mask.to(self.device)

        
        output = self.forward(input_ids,attention_mask)
        loss = self.criterion(output, target).unsqueeze(0)

        return {
            "loss": loss,
        }
    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        target = batch[2]
        # data, target = batch
  
        input_ids.to(self.device)
        attention_mask.to(self.device)

        output = self.forward(input_ids, attention_mask)
        loss = self.criterion(output, target).unsqueeze(0)

        # calculate accuracy
        predicted_classes = torch.argmax(output, dim=1)
        accuracy = torch.sum(predicted_classes == target) / float(len(target))

        # compute confusion matrix
        predicted_classes = predicted_classes.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        self.confusion.add(predicted_classes, target)

        return {
            "loss": loss,
            "val_acc": accuracy
        }
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        # c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader, self.train_dataset, #collate_fn=c_fn,
        )

    def val_dataloader(self):
        # c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            # collate_fn=c_fn,
        )

    def test_dataloader(self):
        # c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader, self.test_dataset, #collate_fn=c_fn,
        )