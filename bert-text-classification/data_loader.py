# coding:utf-8

import io
import random
import numpy as np
from bert import tokenization

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(max_seq_length, tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Create input mask
    input_mask = [1] * len(input_ids)
    # Padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def read_corpus(input_path, vocab_file, max_seq_length):
    texts = []
    labels = []
    input_ids = []
    input_mask = []
    segment_ids = []
    with io.open(input_path, mode = 'r', encoding = 'utf-8') as reader:
        #lines = reader.read().splitlines()
        for line in reader:
            line = line.strip().split('\t')
            if len(line) == 2:
                texts.append(line[0])
                labels.append(line[1])
            elif len(line) == 1:
                texts.append(line[0])
                labels.append(-1)
    tokenizer = tokenization.FullTokenizer(vocab_file = vocab_file)
    for text in texts:
        sentence_input_id, sentence_input_mask, sentence_segment_id = convert_single_example(max_seq_length, 
                tokenizer, text)
        input_ids.append(sentence_input_id)
        input_mask.append(sentence_input_mask)
        segment_ids.append(sentence_segment_id)
    input_ids = np.asarray(input_ids, dtype = np.int32)
    input_mask = np.asarray(input_mask, dtype = np.int32)
    segment_ids = np.asarray(segment_ids, dtype = np.int32)
    labels = np.asarray(labels, dtype = np.int32)
    return input_ids, input_mask, segment_ids, labels


def batch_yield(input_ids, input_mask, segment_ids, 
        labels, batch_size, shuffle):
    if shuffle:
        idx = range(len(labels))
        random.shuffle(idx)
        input_ids = input_ids[idx]
        input_mask = input_mask[idx]
        segment_ids = segment_ids[idx]
        labels = labels[idx]

    size = len(labels)
    batch_num = int((size - 1) / batch_size) + 1
    for i in range(batch_num):
        start_idx = i * batch_size
        end_idx = min(size, (i + 1) * batch_size)
        yield input_ids[start_idx : end_idx], input_mask[start_idx : end_idx], segment_ids[start_idx : end_idx], labels[start_idx : end_idx]

