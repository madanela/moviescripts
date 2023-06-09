import torch
import torch.nn as nn
from transformers import BertForPreTraining,BertForSequenceClassification
from sentence_transformers import SentenceTransformer



class BertSentenceClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertSentenceClassifier, self).__init__()

        # self.bert_freezed =  BertForPreTraining.from_pretrained('distilbert-base-uncased') #BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification")
        self.bert_training = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification").bert
        #BertForPreTraining.from_pretrained('distilbert-base-uncased') #BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type="multi_label_classification")
        #BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification")
        for param in self.bert_training.parameters():
            param.requires_grad = False
        self.dropout_rate = 0.1
        self.lin1 = nn.Linear(768, 256)
        self.lin_layers = nn.ModuleList([nn.Linear(256, 256) for i in range(1)])
        self.lin2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        # bert_1 = self.bert_freezed(input_ids=input_ids, attention_mask=attention_mask)
        bert_2 = self.bert_training(input_ids=input_ids, attention_mask=attention_mask)

        x = nn.functional.relu(self.lin1(bert_2.pooler_output))

        x = nn.functional.dropout(x, self.dropout_rate)

        for lin_layer in self.lin_layers:

            x = nn.functional.relu(lin_layer(x))
            x = nn.functional.dropout(x, self.dropout_rate)

        x = self.lin2(x)
        return x

class BertSentenceClassifier_MeanSentence(nn.Module):
    def __init__(self, num_labels,in_channels,dropout = .3,depth = 3):
        super(BertSentenceClassifier_MeanSentence, self).__init__()

        self.dropout_rate = dropout
        self.lin1 = nn.Linear(in_channels, 256)
        self.lin_layers = nn.ModuleList([nn.Linear(256, 256) for i in range(depth)])
        self.lin2 = nn.Linear(256, num_labels)

    def forward(self, text):
        # bert_1 = self.bert_freezed(input_ids=input_ids, attention_mask=attention_mask)

        x = nn.functional.relu(self.lin1(text))

        x = nn.functional.dropout(x, self.dropout_rate)

        for lin_layer in self.lin_layers:

            x = nn.functional.relu(lin_layer(x))
            x = nn.functional.dropout(x, self.dropout_rate)

        x = self.lin2(x)
        x = nn.functional.softmax(x)
        return x