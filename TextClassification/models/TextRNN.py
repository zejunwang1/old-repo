# coding: UTF-8
import torch
import torch.nn as nn

import numpy as np

class Config(object):
    def __init__(self, data_dir):
        self.model_name = "TextRNN"
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
        self.hidden_size = 128
        self.num_layers = 2

class Model(nn.Module):
    def __init__(self, config, embeddings):
        super(Model, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.max_len = config.pad_size
        self.device = config.device
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
    
    def bi_fetch(self, lstm_outs, seq_lens, batch_size):
        lstm_outs = lstm_outs.view(batch_size, self.max_len, 2, -1)
        
        idx1 = torch.LongTensor([0]).to(self.device)
        idx2 = torch.LongTensor([1]).to(self.device)

        fw_out = torch.index_select(lstm_outs, 2, idx1)
        fw_out = fw_out.view(batch_size * self.max_len, -1)
        bw_out = torch.index_select(lstm_outs, 2, idx2)
        bw_out = bw_out.view(batch_size * self.max_len, -1)

        batch_range = torch.LongTensor(range(batch_size)).to(self.device) * self.max_len
        batch_zeros = torch.zeros(batch_size).long().to(self.device)
        
        fw_index = batch_range + seq_lens.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        out = torch.cat([fw_out, bw_out], dim=1)
        return out

    def forward(self, x):
        out = self.embedding(x[0])
        out, _ = self.lstm(out)
        #out = self.fc(out[:, -1, :])
        batch_size = len(x[1])
        out = self.bi_fetch(out, x[1], batch_size)
        out = self.fc(out)
        return out
