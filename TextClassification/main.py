# coding: UTF-8
import os
import time
import utils
import torch
import jieba
import argparse
import numpy as np
import pickle as pkl
from importlib import import_module
from train_eval import train, init_network

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--mode', type=str, required=True, help='Train or Eval')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == "__main__":
    data_dir = "/home/wangzejun/nlp-tools/TextClassification/data"
    #embedding_path = "/home/wangzejun/nlp-tools/TextClassification/data/tencent_char_embeddings.txt"
    embedding_path = "/home/wangzejun/nlp-tools/TextClassification/data/embedding_Tencent.npz"
    if args.word:
        tokenizer = lambda a : jieba.lcut(a)
    else:
        #tokenizer = lambda a : utils.tokenize(a)
        tokenizer = lambda a: [b for b in a]
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(data_dir)
    use_ngram = False
    if model_name == "NgramLinear":
        use_ngram = True

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    print("Loading vocabulary and pre-trained embeddings...")
    start_time = time.time()
    if os.path.isfile(config.vocab_path):
        #vocab = utils.load_vocab(config.vocab_path)
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = utils.build_vocab(config.train_path, tokenizer=tokenizer, max_size=config.MAX_VOCAB_SIZE, min_freq=config.min_freq)
        utils.save_vocab(vocab, config.vocab_path)
    print(f"Vocab size: {len(vocab)}")
    config.vocab_size = len(vocab)
    config.pad_id = vocab[utils.PAD]
    if args.embedding == "random" or model_name == "NgramLinear":
        embeddings = None
    else:
        #embeddings = torch.tensor(utils.load_vecs(embedding_path, vocab, config.embedding_dim).astype("float32"))
        embeddings = torch.tensor(np.load(embedding_path)["embeddings"].astype("float32"))
    time_dif = utils.get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = x.Model(config, embeddings).to(config.device)
    init_network(model)
    print(model.parameters)

    if args.mode == "Train":
        print("Loading data...")
        start_time = time.time()
        if use_ngram:
            train_data = utils.load_dataset_ngram(config.train_path, vocab, tokenizer, config.pad_size, config.buckets)
            dev_data = utils.load_dataset_ngram(config.dev_path, vocab, tokenizer, config.pad_size, config.buckets)
        else:
            train_data = utils.load_dataset(config.train_path, vocab, tokenizer, config.pad_size)
            dev_data = utils.load_dataset(config.dev_path, vocab, tokenizer, config.pad_size)
        train_iter = utils.build_iterator(train_data, config, use_ngram)
        dev_iter = utils.build_iterator(dev_data, config, use_ngram)
        time_dif = utils.get_time_dif(start_time)
        print("Time usage:", time_dif)
        train(config, model, train_iter, dev_iter)
    else:
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        sentence = input("请输入文本:\n")
        seq_id, seq_len = utils.sample_to_id(sentence, vocab, tokenizer, config.pad_size)
        seq_id = torch.LongTensor([seq_id]).to(config.device)
        seq_len = torch.LongTensor([seq_len]).to(config.device)
        seq_id_len = (seq_id, seq_len)
        with torch.no_grad():
            output = model(seq_id_len)
            label = torch.max(output.data, 1)[1].tolist()
            print("分类结果:" + config.class_list[label[0]])
