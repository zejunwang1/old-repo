import tensorflow as tf
from textcnn import TextCNN
from textrnn import TextRNN
from sklearn.model_selection import train_test_split
from utils import str2bool, get_logger
import os, argparse, time, random
import data

# session configuration
# use the first block of gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# display warning and error info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# dynamic allocation memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# constant allocation memory
# config.gpu_options.per_process_gpu_memory_fraction = 0.3

# hyperparameters
parser = argparse.ArgumentParser(description='CNN/BiLSTM/BiGRU for Text Classification')
parser.add_argument('--train_data', type = str, default = 'data_path', help = 'train data path')
parser.add_argument('--test_data', type = str, default = 'data_path', help = 'test data path')
parser.add_argument('--vocab_data', type = str, default = 'data_path', help = 'vocabulary data path')
parser.add_argument('--category_data', type = str, default = 'data_path', help = 'category data path')
parser.add_argument('--embedding_data', type = str, default = 'data_path', help = 'embedding data path')
parser.add_argument('--batch_size', type = int, default = 64, help = 'sample of each minibatch')
parser.add_argument('--epoch', type = int, default = 10, help = 'epoch of training')
parser.add_argument('--num_filters', type = int, default = 128, help = 'number of filters')
parser.add_argument('--filter_sizes', type = list, default = [3, 4, 5], help = 'filter size')
parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--clip_grad', type = float, default = 5.0, help = 'gradient clipping')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout keep prob')
parser.add_argument('--min_count', type = int, default = 1, help = 'vocabulary min word count')
parser.add_argument('--sequence_length', type = int, default = 50, help = 'maximal sequence length')
parser.add_argument('--update_embedding', type = str2bool, default = True, help = 'update embedding during training')
parser.add_argument('--pretrain_embedding', type = str2bool, default = True, help = 'use pretrained word embedding or init it randomly')
parser.add_argument('--embedding_dim', type = int, default = 100, help = 'word embedding dimension')
parser.add_argument('--ngram', type = int, default = 1, help = '0:character 1:unigram 2:bigram 3:trigram')
parser.add_argument('--memory_size', type = float, default = 4.0, help = 'maximal memory usage for vocabulary')
parser.add_argument('--shuffle', type = str2bool, default = True, help = 'shuffle training data before each epoch')
parser.add_argument('--log_batch_num', type = int, default = 100, help = 'print log info')
parser.add_argument('--binary', type = str2bool, default = False, help = 'pretrained embedding format')
parser.add_argument('--l2_lambda', type = float, default = 0.001, help = 'l2 regularization parameter')
parser.add_argument('--model_name', type = str, default = 'CNN', help = 'CNN/BiLSTM/BiGRU')
parser.add_argument('--mode', type = str, default = 'demo', help = 'train/test/demo')
args = parser.parse_args()


# paths setting
paths = {}
if (args.model_name).lower() == 'cnn':
	tensorboard_dir = os.path.join('.', 'save', 'tensorboard', 'cnn')
	model_dir = os.path.join('.', 'save', 'checkpoints', 'cnn')
	result_dir = os.path.join('.', 'save', 'results', 'cnn')
elif (args.model_name).lower() == 'bilstm':
	tensorboard_dir = os.path.join('.', 'save', 'tensorboard', 'bilstm')
	model_dir = os.path.join('.', 'save', 'checkpoints', 'bilstm')
	result_dir = os.path.join('.', 'save', 'results', 'bilstm')
elif (args.model_name).lower() == 'bigru':
	tensorboard_dir = os.path.join('.', 'save', 'tensorboard', 'bigru')
	model_dir = os.path.join('.', 'save', 'checkpoints', 'bigru')
	result_dir = os.path.join('.', 'save', 'results', 'bigru')
if not os.path.exists(tensorboard_dir):
	os.makedirs(tensorboard_dir)
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
if not os.path.exists(result_dir):
	os.makedirs(result_dir)

model_prefix = os.path.join(model_dir, 'best_model')
log_path = os.path.join(result_dir, 'log.txt')
result_path = os.path.join(result_dir, 'results.txt')
paths['summary_path'] = tensorboard_dir
paths['model_path'] = model_prefix
paths['log_path'] = log_path
paths['result_path'] = result_path



# read corpus and get training data
if args.mode == 'train':
	train_path = os.path.join('.', args.train_data, 'train_data')
	# build training vocabulary and category
	if args.ngram == 0:
		vocab, category = data.build_character_vocab(train_path)
	elif args.ngram == 1 or args.ngram == 2 or args.ngram == 3:
		vocab, category = data.build_ngram_vocab(train_path, args.ngram, args.min_count, args.memory_size)
	else:
		raise ValueError('Unknown ngram value!')
	# save vocabulary and category
	vocab_path = os.path.join('.', args.vocab_data, 'vocab_data')
	data.vocab_dump(vocab, vocab_path)
	category_path = os.path.join('.', args.category_data, 'category_data')
	data.category_dump(category, category_path)
	# get embeddings
	if args.pretrain_embedding:
		embedding_path = os.path.join('.', args.embedding_data, 'sina.news.vec')
		embeddings = data.pretrained_embedding(vocab, embedding_path, args.embedding_dim, args.binary)
	else:
		embeddings = data.random_embedding(vocab, args.embedding_dim)
	# training data generation
	x, y = data.read_corpus(train_path, vocab, category, args.ngram, args.sequence_length)
	# training and validation data split
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)
	# training model
	if (args.model_name).lower() == 'cnn':
		model = TextCNN(args, embeddings, category, paths, config)
		model.build_graph()
		model.train(x_train, y_train, x_val, y_val)
	elif (args.model_name).lower() == 'bilstm' or (args.model_name).lower() == 'bigru':
		model = TextRNN(args, embeddings, category, paths, config)
		model.build_graph()
		model.train(x_train, y_train, x_val, y_val)
elif args.mode == 'test':
	test_path = os.path.join('.', args.test_data, 'test_data')
	vocab_path = os.path.join('.', args.vocab_data, 'vocab_data')
	vocab = data.vocab_load(vocab_path)
	category_path = os.path.join('.', args.category_data, 'category_data')
	category = data.category_load(category_path)
	# get embeddings
	if args.pretrain_embedding:
		embedding_path = os.path.join('.', args.embedding_data, 'sina.news.vec')
		embeddings = data.pretrained_embedding(vocab, embedding_path, args.embedding_dim, args.binary)
	else:
		embeddings = data.random_embedding(vocab, args.embedding_dim)
	x_test, y_test = data.read_corpus(test_path, vocab, category, args.ngram, args.sequence_length)
	ckpt_file = tf.train.latest_checkpoint(model_dir)
	paths['model_path'] = ckpt_file
	if (args.model_name).lower() == 'cnn':
		model = TextCNN(args, embeddings, category, paths, config)
		model.build_graph()
		y_pred, y_prob_mat = model.test(x_test)
		model.save_predictions(y_pred, y_prob_mat)
	elif (args.model_name).lower() == 'bilstm' or (args.model_name).lower() == 'bigru':
		model = TextRNN(args, embeddings, category, paths, config)
		model.build_graph()
		y_pred, y_prob_mat = model.test(x_test)
		model.save_predictions(y_pred, y_prob_mat)
