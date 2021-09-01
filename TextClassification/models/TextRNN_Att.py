# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Config(object):
    def __init__(self, data_dir):
        self.model_name = "TextRNN_Att"
        self.train_path = data_dir + "/train.txt"
        self.dev_path = data_dir + "/dev.txt"
        self.class_list = [x.strip() for x in open(data_dir + "/class.txt", "r", encoding="UTF-8").readlines()]
        #self.vocab_path = data_dir + "/vocab.txt"
        self.vocab_path = data_dir + "/vocab.pkl"
        self.model_path = data_dir + "/model/" + self.model_name + ".bin"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.pad_id = 0
        self.pad_size = 32
        self.vocab_size = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.embedding_dim = 200
        self.MAX_VOCAB_SIZE = 10000
        self.min_freq = 1
        self.num_layers = 2
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64

class Model(nn.Module):
    def __init__(self, config, embeddings):
        super(Model, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size_1, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.w = nn.Parameter(torch.zeros(config.hidden_size_1 * 2))
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size_1 * 2, config.hidden_size_2)
        self.fc2 = nn.Linear(config.hidden_size_2, config.num_classes)
    
    def forward(self, x):
        emb = self.embedding(x[0])
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

