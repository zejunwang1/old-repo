# coding: uTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Config(object):
    def __init__(self, data_dir):
        self.model_name = "DPCNN"
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
        self.num_filters = 250

class Model(nn.Module):
    def __init__(self, config, embeddings):
        super(Model, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.region_conv = nn.Conv2d(1, config.num_filters, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))     # top-bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))     # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def _block(self, x):
        x = self.padding2(x)
        dx = self.maxpool(x)

        x = self.padding1(dx)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # short-cut
        x = x + dx
        return x 

    def forward(self, x):
        x = self.embedding(x[0])
        x = x.unsqueeze(1)
        rx = self.region_conv(x)

        x = self.padding1(rx)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        x = x + rx
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
