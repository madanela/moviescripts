
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



class SentenceClassifierEncoded(pl.LightningModule):
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
        # self.bert_freezed =  BertForPreTraining.from_pretrained('distilbert-base-uncased') #BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification")
        
        # model
        self.model = hydra.utils.instantiate(config.model)

        self.labels_info = dict()
        self.validation_step_outputs = []
        self.validation_step_outputs_acc = []
        self.training_step_outputs = []


    def forward(self, data):
        x = self.model(data)
        return x
    def training_step(self, batch, batch_idx):

        data = batch[0]
        target = batch[1]


        data.to(self.device)
        
        output = self.forward(data)

        loss = self.criterion(output, target).unsqueeze(0)

        target = target.detach().cpu().numpy()


        self.training_step_outputs.append(loss[0])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss[0],
        }
    def validation_step(self, batch, batch_idx):
        data = batch[0]
        target = batch[1]
        # data, target = batch
  
        data.to(self.device)

        output = self.forward(data)

        loss = self.criterion(output, target).unsqueeze(0)
        # print("validation loss : ",loss)
        # calculate accuracy
        predicted_classes = torch.argmax(output, dim=1)
        accuracy = torch.sum(predicted_classes == target) / float(len(target))

        # compute confusion matrix
        predicted_classes = predicted_classes.detach().cpu().numpy()
        # target = target.detach().cpu().numpy()
        
        self.confusion.add(predicted_classes, target)
        self.validation_step_outputs.append(loss[0])
        self.validation_step_outputs_acc.append(accuracy)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            "val_loss": loss[0],
            "val_acc": accuracy
        }
    def on_train_epoch_start(self):
        self.training_step_outputs = []
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
    def on_train_epoch_end(self):
        train_loss = torch.tensor(self.training_step_outputs).mean()
        results = {"train_loss": train_loss}
        self.log_dict(results)
        self.confusion.reset()
    def on_validation_epoch_end(self,**kwargs):
        val_loss = torch.tensor(self.validation_step_outputs).mean()
        val_acc = torch.tensor(self.validation_step_outputs_acc).mean()
        results = {"val_loss": val_loss,"val_acc":val_acc}


        self.log_dict(results)
        self.confusion.reset()
    def test_step(self, batch, batch_idx):
        data = batch[0]
        target = batch[1]
    
        data.to(self.device)

        output = self.forward(data)

        loss = self.criterion(output, target).unsqueeze(0)
        
        # calculate accuracy
        predicted_classes = torch.argmax(output, dim=1)
        accuracy = torch.sum(predicted_classes == target) / float(len(target))

        # compute confusion matrix
        predicted_classes = predicted_classes.detach().cpu().numpy()

        # target = target.detach().cpu().numpy()
        self.confusion.add(predicted_classes, target)

        return {
            "test_loss": loss[0],
            "test_acc": accuracy
        }
    # def test_step(self, batch, batch_idx):
    #     data, target = batch
    #     inverse_maps = data.inverse_maps
    #     original_labels = data.original_labels
    #     data = ME.SparseTensor(coords=data.coordinates, feats=data.features)
    #     data.to(self.device)
    #     output = self.forward(data)
    #     loss = 0
    #     if original_labels[0].size > 0:
    #         loss = self.criterion(output.F, target).unsqueeze(0)
    #         target = target.detach().cpu()
    #     original_predicted = []
    #     for i in range(len(inverse_maps)):
    #         # https://github.com/NVIDIA/MinkowskiEngine/issues/119
    #         out = output.F[output.C[:, 0] == i]
    #         out = out.max(1)[1].view(-1).detach().cpu()
    #         original_predicted.append(out[inverse_maps[i]].numpy())

    #     return {
    #         "test_loss": loss,
    #         "metrics": {
    #             "original_output": original_predicted,
    #             "original_label": original_labels,
    #         },
    #     }
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
    







#########################################
#             OLD RUN                   #
#########################################



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