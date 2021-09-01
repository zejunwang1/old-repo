# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self, data_dir):
        self.model_name = "TextCNN"
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
        self.num_epochs = 20
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.embedding_dim = 200
        self.MAX_VOCAB_SIZE = 10000
        self.min_freq = 1
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

class Model(nn.Module):
    def __init__(self, config, embeddings=None):
        super(Model, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
