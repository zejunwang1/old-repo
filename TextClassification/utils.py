# coding: UTF-8
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import timedelta

PAD, UNK = "<PAD>", "<UNK>"

# 分字
def tokenize(text):
    chars = []
    eng = []
    eng_flag = 0
    for char in text:
        uc = ord(char)
        if ((uc >= 0x4E00 and uc <= 0x9FFF) or
            (uc >= 0x3400 and uc <= 0x4DBF) or
            (uc >= 0x20000 and uc <= 0x2A6DF) or
            (uc >= 0x2A700 and uc <= 0x2B73F) or
            (uc >= 0x2B740 and uc <= 0x2B81F) or
            (uc >= 0x2B820 and uc <= 0x2CEAF) or
            (uc >= 0xF900 and uc <= 0xFAFF) or
            (uc >= 0x2F800 and uc <= 0x2FA1F)):
            if eng_flag:
                eng_word = "".join(eng)
                chars.append(eng_word)
                eng_flag = 0
                eng = []
            chars.append(char)
        elif ((uc >= 97 and uc <= 122) or
              (uc >= 65 and uc <= 90) or
              (uc >= 48 and uc <= 57)):
              eng_flag = 1
              eng.append(char)
        else:
            if eng_flag:
                eng_word = "".join(eng)
                chars.append(eng_word)
                eng_flag = 0
                eng = []
            chars.append(char)
    if eng_flag:
        eng_word = "".join(eng)
        chars.append(eng_word)
    return chars

def load_vocab(vocab_path):
    vocab_dic = {}
    idx = 0
    with open(vocab_path, "r", encoding="UTF-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            if word in vocab_dic:
                continue
            vocab_dic[word] = idx
            idx += 1
        if UNK not in vocab_dic:
            vocab_dic.update({UNK: len(vocab_dic)})
        if PAD not in vocab_dic:
            vocab_dic.update({PAD: len(vocab_dic)})
    return vocab_dic

def save_vocab(vocab, vocab_path):
    f = open(vocab_path, "w", encoding="UTF-8")
    for key in vocab.keys():
        f.write(key)
        f.write("\n")
    f.close()

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        if UNK not in vocab_dic:
            vocab_dic.update({UNK: len(vocab_dic)})
        if PAD not in vocab_dic:
            vocab_dic.update({PAD: len(vocab_dic)})
    return vocab_dic

def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets

def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

def load_dataset(file_path, vocab, tokenizer, pad_size=32):
    contents = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split("\t")
            word_ids = []
            tokens = tokenizer(content)
            seq_len = len(tokens)
            if pad_size:
                if len(tokens) < pad_size:
                    tokens.extend([PAD] * (pad_size - len(tokens)))
                else:
                    tokens = tokens[:pad_size]
                    seq_len = pad_size
            # word to id
            for token in tokens:
                word_ids.append(vocab.get(token, vocab.get(UNK)))
            contents.append((word_ids, int(label), seq_len))
    return contents

def load_dataset_ngram(file_path, vocab, tokenizer, pad_size=32, buckets=250499):
    contents = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split("\t")
            word_ids = []
            tokens = tokenizer(content)
            seq_len = len(tokens)
            if pad_size:
                if len(tokens) < pad_size:
                    tokens.extend([PAD] * (pad_size - len(tokens)))
                else:
                    tokens = tokens[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in tokens:
                word_ids.append(vocab.get(word, vocab.get(UNK)))
            # ngram to id
            bigram = []
            trigram = []
            for i in range(pad_size):
                bigram.append(biGramHash(word_ids, i, buckets))
                trigram.append(triGramHash(word_ids, i, buckets))
            contents.append((word_ids, int(label), seq_len, bigram, trigram))
    return contents

def sample_to_id(text, vocab, tokenizer, pad_size=32):
    text = text.strip()
    tokens = tokenizer(text)
    seq_len = len(tokens)
    seq_id = []
    if pad_size:
        if len(tokens) < pad_size:
            tokens.extend([PAD] * (pad_size - len(tokens)))
        else:
            tokens = tokens[:pad_size]
            seq_len = pad_size
    # word to id
    for token in tokens:
        seq_id.append(vocab.get(token, vocab.get(UNK)))
    return seq_id, seq_len

class DatasetIterater(object):
    def __init__(self, samples, batch_size, device, use_ngram=False):
        self.samples = samples
        self.batch_size = batch_size
        self.n_batches = len(samples) // batch_size
        self.residue = False
        self.use_ngram = use_ngram
        if len(samples) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
    
    def _to_tensor(self, batch_data):
        x = torch.LongTensor([_[0] for _ in batch_data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch_data]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in batch_data]).to(self.device)
        if self.use_ngram:
            bigram = torch.LongTensor([_[3] for _ in batch_data]).to(self.device)
            trigram = torch.LongTensor([_[4] for _ in batch_data]).to(self.device)
            return (x, seq_len, bigram, trigram), y
        else:
            return (x, seq_len), y
    
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.samples[self.index * self.batch_size: len(self.samples)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.samples[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config, use_ngram=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, use_ngram)
    return iter

def load_vecs(file_path, vocab, embedding_dim):
    embeddings = 2 * (np.random.rand(len(vocab), embedding_dim) - 0.5)
    f = open(file_path, "r", encoding="UTF-8")
    for _, line in enumerate(f.readlines()):
        line = line.strip().split(" ")
        if len(line) != embedding_dim + 1:
            continue
        word = line[0]
        if word in vocab:
            idx = vocab[word]
            vec = [float(x) for x in line[1:]]
            embeddings[idx] = np.asarray(vec, dtype="float32")
    f.close()
    pad_id = vocab[PAD]
    vec = [0.0] * embedding_dim
    embeddings[pad_id] = np.asarray(vec, dtype="float32")
    return embeddings

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
