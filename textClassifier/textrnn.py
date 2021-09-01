import tensorflow as tf
import numpy as np
import sys, time
from data import batch_yield
from utils import get_logger
from tensorflow.contrib.rnn import LSTMCell, GRUCell

class TextRNN(object):
	def __init__(self, args, embeddings, category, paths, config):
		self.batch_size = args.batch_size
		self.epoch_num = args.epoch
		self.log_batch = args.log_batch_num
		self.embeddings = embeddings
		self.update_embedding = args.update_embedding
		self.dropout_keep_prob = args.dropout
		self.optimizer = args.optimizer
		self.learning_rate = args.learning_rate
		self.clip_grad = args.clip_grad
		self.shuffle = args.shuffle
		self.l2_lambda = args.l2_lambda
		self.sequence_length = args.sequence_length
		self.num_classes = len(category)
		self.num_filters = args.num_filters
		self.embedding_dim = args.embedding_dim
		self.hidden_size = args.embedding_dim
		self.filter_sizes = args.filter_sizes
		self.model_name = args.model_name
		self.summary_path = paths['summary_path']
		self.model_path = paths['model_path']
		self.result_path = paths['result_path']
		self.config = config
		self.logger = get_logger(paths['log_path'])

	def build_graph(self):
		self.add_placeholders()
		self.lookup_layer_op()
		self.BiRNN_layer_op()
		self.softmax_layer_op()
		self.loss_op()
		self.trainstep_op()
		self.init_op()

	def add_placeholders(self):
		self.word_ids = tf.placeholder(dtype = tf.int32, shape = [None, self.sequence_length], name = 'word_ids')
		self.labels = tf.placeholder(dtype = tf.float32, shape = [None, self.num_classes], name = 'labels')
		self.dropout_kp = tf.placeholder(dtype = tf.float32, shape = [], name = 'dropout_kp')

	def lookup_layer_op(self):
		with tf.variable_scope('words'):
			_word_embeddings = tf.Variable(self.embeddings,
										   dtype = tf.float32,
										   trainable = self.update_embedding,
										   name = '_word_embeddings')
			word_embeddings = tf.nn.embedding_lookup(params = _word_embeddings,
													 ids = self.word_ids,
													 name = 'word_embeddings')
			self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_kp)

	def BiRNN_layer_op(self):
		with tf.variable_scope('bi-rnn'):
			if (self.model_name).lower() == 'bilstm':
				cell_fw = LSTMCell(self.hidden_size)
				cell_bw = LSTMCell(self.hidden_size)
			elif (self.model_name).lower() == 'bigru':
				cell_fw = GRUCell(self.hidden_size)
				cell_bw = GRUCell(self.hidden_size)
			else:
				raise ValueError('Unknown model name!')
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings, dtype = tf.float32)
			output_rnn = tf.concat(outputs, axis = 2)
			self.output_rnn_last = output_rnn[:, -1, :]

	def softmax_layer_op(self):
		with tf.variable_scope('softmax'):
			W = tf.get_variable(name = 'W',
								shape = [self.hidden_size * 2, self.num_classes],
								initializer = tf.contrib.layers.xavier_initializer())
			b = tf.get_variable(name = 'b', shape = [self.num_classes])
			self.scores = tf.nn.xw_plus_b(self.output_rnn_last, W, b, name = 'scores')
			self.normalized_scores = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.scores, 1, name = 'predictions')

	def loss_op(self):
		losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.labels)
		self.loss = tf.reduce_mean(losses)
		correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name = 'accuracy')
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('accuracy', self.accuracy)

	def trainstep_op(self):
		with tf.variable_scope('train_step'):
			self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
			if self.optimizer == 'Adam':
				optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
			elif self.optimizer == 'Adadelta':
				optim = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
			elif self.optimizer == 'Adagrad':
				optim = tf.train.AdagradOptimizer(learning_rate = self.learning_rate)
			elif self.optimizer == 'RMSProp':
				optim = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
			elif self.optimizer == 'Momentum':
				optim = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
			elif self.optimizer == 'SGD':
				optim = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
			else:
				optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

			grads_and_vars = optim.compute_gradients(self.loss)
			grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
			self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step = self.global_step)

	def init_op(self):
		self.init_op = tf.global_variables_initializer()

	def add_summary(self, sess):
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

	def get_feed_dict(self, x, y, dropout):
		feed_dict = {
			self.word_ids : x,
			self.labels : y,
			self.dropout_kp : dropout
		}
		return feed_dict

	def train(self, x_train, y_train, x_val, y_val):
		saver = tf.train.Saver(tf.global_variables())

		with tf.Session(config = self.config) as sess:
			sess.run(self.init_op)
			self.add_summary(sess)

			best_val_acc = 0.0
			num_batches = int((len(x_train) - 1) / (self.batch_size)) + 1
			for epoch in range(self.epoch_num):
				batch_num = 0
				sys.stdout.write('Epoch : ' + str(epoch + 1) + '\n')
				batch_train = batch_yield(x_train, y_train, self.batch_size, self.shuffle)
				for x_batch, y_batch in batch_train:
					sys.stdout.write('Processing : {} batch / {} batches.\n'.format(batch_num + 1, num_batches))
					feed_dict = self.get_feed_dict(x_batch, y_batch, self.dropout_keep_prob)
					step_num = epoch * num_batches + batch_num + 1

					if batch_num % (self.log_batch) == 0:
						feed_dict[self.dropout_kp] = 1.0
						train_loss, train_acc = sess.run([self.loss, self.accuracy], feed_dict = feed_dict)
						val_loss, val_acc = self.evaluate(sess, x_val, y_val)
						self.logger.info(
							'Iter: {0:>1}, Batch Num: {1:>1}, Train Loss: {2:>1.2}, Train Acc: {3:>2.2%}, Val Loss: {4:>1.2}, Val Acc: {5:>2.2%}'.format(
								epoch + 1, batch_num + 1, train_loss, train_acc, val_loss, val_acc))

					_, train_loss, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
																 feed_dict = feed_dict)
					self.file_writer.add_summary(summary, step_num)
					batch_num += 1

				train_loss, train_acc = self.evaluate(sess, x_train, y_train)
				val_loss, val_acc = self.evaluate(sess, x_val, y_val)

				self.logger.info('*********************************************************************************')
				self.logger.info(
					'Iter: {0:>1}, Train Loss: {1:>1.2}, Train Acc: {2:>2.2%}, Val Loss: {3:>1.2}, Val Acc: {4:>2.2%}'.format(
						epoch + 1, train_loss, train_acc, val_loss, val_acc))
				self.logger.info('*********************************************************************************\n')

				if val_acc > best_val_acc:
					best_val_acc = val_acc
					saver.save(sess = sess, save_path = self.model_path, global_step = step_num)

	def evaluate(self, sess, x, y):
		batches = batch_yield(x, y, self.batch_size, self.shuffle)
		total_loss, total_acc = 0.0, 0.0
		for x_batch, y_batch in batches:
			batch_len = len(x_batch)
			feed_dict = self.get_feed_dict(x_batch, y_batch, 1.0)
			loss, acc = sess.run([self.loss, self.accuracy], feed_dict = feed_dict)
			total_loss += loss * batch_len
			total_acc += acc * batch_len

		total_len = len(x)
		return total_loss / total_len, total_acc / total_len

	def test(self, x_test):
		saver = tf.train.Saver()
		with tf.Session(config = self.config) as sess:
			saver.restore(sess = sess, save_path = self.model_path)
			data_len = len(x_test)
			num_batches = int((data_len - 1) / self.batch_size) + 1

			y_pred = np.zeros(shape = data_len, dtype = np.int32)
			y_prob_mat = np.zeros(shape = (data_len, self.num_classes), dtype = np.float32)
			for i in range(num_batches):
				start_id = i * (self.batch_size)
				end_id = min((i + 1) * (self.batch_size), data_len)
				feed_dict = {
					self.word_ids : x_test[start_id : end_id],
					self.dropout_kp : 1.0
				}
				y_pred[start_id : end_id] = sess.run(self.predictions, feed_dict = feed_dict)
				y_prob_mat[start_id : end_id] = sess.run(self.normalized_scores, feed_dict = feed_dict)

		return y_pred, y_prob_mat

	def save_predictions(self, y_pred, y_prob_mat):
		f = open(self.result_path, mode = 'w', encoding = 'utf-8', errors = 'ignore')
		f.write('label' + '\t' + 'probability' + '\n')
		for i in range(len(y_pred)):
			f.write(str(y_pred[i]))
			for j in range(y_prob_mat.shape[1]):
				f.write('\t' + str(y_prob_mat[i, j]))
			f.write('\n')
		f.close()

