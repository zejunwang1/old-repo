import os, sys, six, pickle, random
import numpy as np
import tensorflow.contrib.keras as kr


def get_n_gram(line, pos, gram):
	"""
	get n-gram word in a sentence
	gram: 1-unigram 2-bigram 3-trigram
	"""
	if pos < 0:
		return None
	if pos + gram > len(line):
		return None
	word = line[pos]
	for i in range(1, gram):
		word = word + '@$' + line[pos + i]
	return word


def build_character_vocab(corpus_path):
	"""
	build character level vocabulary
	build category vocabulary
	"""
	vocab = {}
	category = {}
	labels = set()
	num = 0
	with open(corpus_path, mode = 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			if num % 1000 == 0:
				sys.stdout.write('\rProcessed ' + str(num) + ' lines')
				sys.stdout.flush()
			num += 1
			try:
				label, sent = line.strip().split('\t')
				if sent:
					labels.add(label)
					sent = list(''.join(sent.strip().split()))
					for char in sent:
						if char.isdigit():
							char = '<NUM>'
						elif ('\u0041' <= char <='\u005a') or ('\u0061' <= char <='\u007a'):
							word = '<ENG>'
						if char not in vocab:
							vocab[char] = 1
						else:
							vocab[char] += 1
			except:
				pass

	print('\nProcessed ' + str(num) + ' lines')
	idx = 1
	for char in vocab.keys():
		vocab[char] = idx
		idx += 1
	vocab['<PAD>'] = 0
	vocab['<UNK>'] = idx
	category = dict(zip(labels, range(len(labels))))
	print('Vocabulary size: ', len(vocab))
	print('Number of categories: ', len(category))
	return vocab, category


def build_ngram_vocab(corpus_path, ngram, min_count, memory_size):
	"""
	build ngram level vocabulary
	build category vocabulary
	ngram: 1-unigram 2-bigram 3-trigram
	min_count: minimal vocabulary word count
	memory_size: maximal memory usage for vocabulary
	"""
	vocab = {}
	category = {}
	labels = set()
	num = 0
	threshold = 1
	memory_used = 0
	memory_size = float(memory_size) * 1000**3
	with open(corpus_path, mode = 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			if num % 1000 == 0:
				sys.stdout.write('\rProcessed ' + str(num) + ' lines')
				sys.stdout.flush()
			num += 1
			try:
				label, sent = line.strip().split('\t')
				if sent:
					labels.add(label)
					sent = sent.strip().split()
					for i in range(len(sent)):
						for j in range(1, ngram + 1):
							word = get_n_gram(sent, i, j)
							if word is None:
								continue
							if word.isdigit():
								word = '<NUM>'
							if word not in vocab:
								memory_used += sys.getsizeof(word)
								vocab[word] = 1
								if memory_used + sys.getsizeof(vocab) > 0.8 * memory_size:
									threshold += 1
									n = len(vocab)
									vocab = {w: c for w, c in six.iteritems(vocab) if c >= threshold}
									memory_used *= float(len(vocab)) / n 
							else:
								vocab[word] += 1
			except:
				pass

	print('\nProcessed ' + str(num) + ' lines')
	vocab = {w: c for w, c in six.iteritems(vocab) if c >= min_count}
	idx = 1
	for word in vocab.keys():
		vocab[word] = idx
		idx += 1
	vocab['<PAD>'] = 0
	vocab['<UNK>'] = idx
	category = dict(zip(labels, range(0, len(labels))))
	print('Vocabulary size: ', len(vocab))
	print('Number of categories: ', len(category))
	return vocab, category


def vocab_dump(vocab, vocab_path):
	"""
	dump vocabulary
	"""
	with open(vocab_path, mode = 'wb') as fw:
		pickle.dump(vocab, fw)


def vocab_load(vocab_path):
	"""
	load vocabulary
	"""
	vocab_path = os.path.join(vocab_path)
	with open(vocab_path, mode = 'rb') as fr:
		word2id = pickle.load(fr)
	return word2id


def category_dump(category, category_path):
	"""
	save category
	"""
	with open(category_path, mode = 'wb') as fw:
		pickle.dump(category, fw)


def category_load(category_path):
	"""
	load category
	"""
	with open(category_path, mode = 'rb') as fr:
		category = pickle.load(fr)
	return category


def random_embedding(vocab, embedding_dim):
	"""
	generate random embedding matrix
	vocab: vocabulary
	embedding_dim: random embedding vectors dimension
	"""
	embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
	embedding_mat = np.float32(embedding_mat)
	return embedding_mat


def pretrained_embedding(vocab, embedding_path, embedding_dim, binary):
	"""
	load pretrained embedding vectors
	generate vocabulary embedding matrix
	vocab: vocabulary
	embedding_path: pretrained embedding vectors path
	embedding_dim: pretrained embedding vectors dimension
	binary: pretrained embedding vectors format
			True - binary format
			False - non-binary format 
	"""
	word_vecs = {}
	num = 0
	if binary:
		with open(embedding_path, mode = 'rb', encoding = 'utf-8', errors = 'ignore') as fr:
			header = f.readline()
			size, dim = map(int, header.strip().split())
			if dim != embedding_dim:
				raise ValueError('embedding dimension does not match')
			binary_len = np.dtype('float32').itemsize * dim
			for num in range(size):
				if num % 100 == 0:
					sys.stdout.write('\rProcessed ' + str(num) + ' pretrained vectors')
					sys.stdout.flush()
				word = []
				while True:
					ch = fr.read(1)
					if ch == ' ':
						word = ''.join(word)
						break
					if ch != '\n':
						word.append(ch)
				if word in vocab:
					word_vecs[word] = np.fromstring(fr.read(binary_len), dtype = 'float32')
				else:
					f.read(binary_len)
		print('\nProcessed ' + str(num) + ' pretrained vectors')
	else:
		with open(embedding_path, mode = 'r', encoding = 'utf-8', errors = 'ignore') as fr:
			for line in fr:
				if num == 0:
					size, dim = map(int, line.strip().split())
					if dim != embedding_dim:
						raise ValueError('embedding dimension does not match')
				else:
					line = line.strip().split()
					if not(line):
						continue
					word = line[0]
					if word in vocab:
						word_vecs[word] = np.array(line[1:], dtype = 'float32')
				if num % 100 == 0:
					sys.stdout.write('\rProcessed ' + str(num) + ' pretrained vectors')
					sys.stdout.flush()
				num += 1
		print('\nProcessed ' + str(num) + ' pretrained vectors')

	# add random word vectors beyond pretrained vectors
	for word in vocab:
		if word not in word_vecs:
			word_vecs[word] = np.random.uniform(-0.25, 0.25, embedding_dim)

	# generate final embedding matrix
	W = np.zeros(shape = (len(vocab), embedding_dim), dtype = 'float32')
	for word in vocab:
		W[vocab[word]] = word_vecs[word]

	return W


def sentence2id(sent, vocab, ngram):
	"""
	transform a sentence into a list of index
	sent: list of sentence
	vocab: vocabulary
	ngram: n-gram feature in a sentence 
		   0-character 1-unigram 2-bigram 3-trigram
	"""
	sentence_id = []
	for i in range(len(sent)):
		word = sent[i]
		if word.isdigit():
			word = '<NUM>'
		if word not in vocab:
			word = '<UNK>'
		if ngram == 0:
			if ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
				word = '<ENG>'
			if word in vocab:
				sentence_id.append(vocab[word])
		else:
			for j in range(1, ngram + 1):
				ngram_word = get_n_gram(sent, i, j)
				if ngram_word in vocab:
					sentence_id.append(vocab[ngram_word])

	return sentence_id


def read_corpus(corpus_path, vocab, category, ngram, sequence_length):
	"""
	transform original corpus into index representation
	vocab: vocabulary
	ngram: 0-character 1-unigram 2-bigram 3-trigram
	sequence_length: maximal sequence length
	"""
	data_id, label_id = [], []
	num = 0
	with open(corpus_path, mode = 'r', encoding = 'utf-8', errors = 'ignore') as f:
		for line in f:
			if num % 1000 == 0:
				sys.stdout.write('\rProcessed ' + str(num) + ' lines')
				sys.stdout.flush()
			num += 1
			try:
				label, sent = line.strip().split('\t')
				if sent:
					label_id.append(category[label])
					if ngram == 0:
						sent = list(''.join(sent.strip().split()))
					else:
						sent = sent.strip().split()
					sent_id = sentence2id(sent, vocab, ngram)
					data_id.append(sent_id)
			except:
				pass

	print('\nProcessed ' + str(num) + ' lines')
	x_pad = kr.preprocessing.sequence.pad_sequences(data_id, sequence_length)
	y_pad = kr.utils.to_categorical(label_id, num_classes = len(category))
	return x_pad, y_pad


def batch_yield(x, y, batch_size, shuffle):
	"""
	generate batches
	"""
	if shuffle:
		idx = range(len(y))
		random.shuffle(idx)
		x = x[idx]
		y = y[idx]

	size = len(x)
	batch_num = int((size - 1) / batch_size) + 1
	for i in range(batch_num):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, size)
		yield x[start_id : end_id], y[start_id : end_id]
	
