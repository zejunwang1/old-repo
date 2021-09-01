# coding:utf-8

import numpy as np
import tensorflow as tf
from bert import modeling
from data_loader import batch_yield

class BertTextClassification(object):
    def __init__(self, bert_config, batch_size, max_seq_length, is_shuffle, optimizer, learning_rate, max_epoches, init_checkpoint, clip_grad, num_labels, is_training, use_one_hot_embeddings, paths):
        self.bert_config = bert_config
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.optimizer = optimizer
        self.is_shuffle = is_shuffle
        self.learning_rate = learning_rate
        self.max_epoches = max_epoches
        self.clip_grad = clip_grad
        self.num_labels = num_labels
        self.is_training = is_training
        self.init_checkpoint = init_checkpoint
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.model_path = paths["model_path"]
        self.log_path = paths["log_path"]
    
    def build(self):
        self.add_placeholders()
        self.create_model()
        self.train_step()
        self.load_model()

    def add_placeholders(self):
        self.input_ids = tf.placeholder(shape = [None, self.max_seq_length], dtype = tf.int32, name = "input_ids")
        self.input_mask = tf.placeholder(shape = [None, self.max_seq_length], dtype = tf.int32, name = "input_mask")
        self.segment_ids = tf.placeholder(shape = [None, self.max_seq_length], dtype = tf.int32, name = "segment_ids")
        self.input_labels = tf.placeholder(shape = [self.batch_size], dtype = tf.int32, name = "input_labels")
    
    def create_model(self):
        model = modeling.BertModel(
            config = self.bert_config,
            is_training = self.is_training,
            input_ids = self.input_ids,
            input_mask = self.input_mask,
            token_type_ids = self.segment_ids,
            use_one_hot_embeddings = self.use_one_hot_embeddings)

        output_tokens_layer = model.get_sequence_output()

        output_sentence_layer = model.get_pooled_output()

        hidden_size = output_sentence_layer.shape[-1].value

        output_weights = tf.get_variable(
                name = "output_weights", 
                shape = [self.num_labels, hidden_size],
                initializer = tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
                name = "output_bias",
                shape = [self.num_labels],
                initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if self.is_training:
                output_sentence_layer = tf.nn.dropout(output_sentence_layer, keep_prob = 0.9)
            logits = tf.matmul(output_sentence_layer, output_weights, 
                    transpose_b = True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis = -1)
            probabilities = tf.nn.softmax(logits, axis = -1)
            one_hot_labels = tf.one_hot(self.input_labels, 
                    depth = self.num_labels, dtype = tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis = -1)
            loss = tf.reduce_mean(per_example_loss)
            predict = tf.argmax(tf.nn.softmax(logits), axis = 1)
            acc = tf.reduce_mean(tf.cast(tf.equal(self.input_labels, tf.cast(predict, dtype = tf.int32)), "float"), name = "accuracy")
            self.loss = loss
            self.acc = acc
            self.prediction = predict
            self.softmax_probabilities = probabilities

    def train_step(self):
        '''
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
            if self.optimizer == "Adam":
                optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            elif self.optimizer == "Adadelta":
                optim = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
            elif self.optimizer == "Adagrad":
                optim = tf.train.AdagradOptimizer(learning_rate = self.learning_rate)
            elif self.optimizer == "RMSProp":
                optim = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
            elif self.optimizer == "Momentum":
                optim = tf.train.MomentumOptimizer(learning_rate = self.learning_rate)
            elif self.optimizer == "SGD":
                optim = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
            else:
                optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step = self.global_step)
        '''
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def load_model(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        # Print the parameters of the pretrained bert model
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name,
                    var.shape, init_string)

    def train(self, input_ids_train, input_mask_train, segment_ids_train,
            input_labels_train):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_batches = int((len(input_labels_train) - 1) / self.batch_size) + 1
            for epoch in range(self.max_epoches):
                batch_idx = 0
                print("Epoch : " + str(epoch + 1) + "\n")
                '''
                batch_train = batch_yield(input_ids_train, 
                                          input_mask_train,
                                          segment_ids_train, 
                                          input_labels_train,
                                          self.batch_size,
                                          self.is_shuffle)
                for input_ids_batch, input_mask_batch, segment_ids_batch, input_labels_batch in batch_train:
                    print("Processing : {} batch / {} batches.\n".format(batch_idx + 1, num_batches))
                    feed_dict = {self.input_ids : input_ids_batch,
                                 self.input_mask : input_mask_batch,
                                 self.segment_ids : segment_ids_batch,
                                 self.input_labels : input_labels_batch}
                    _, batch_loss, batch_acc = sess.run([self.train_op, self.loss, self.acc], feed_dict = feed_dict)
                    print("\tbatch_acc : {}".format(batch_acc))
                    batch_idx += 1
                
                train_loss, train_acc = self.eval(sess, input_ids_train, 
                        input_mask_train, segment_ids_train, input_labels_train)
                print("\ttrain_acc : {}".format(train_acc))
                '''
                shuffIndex = np.random.permutation(np.arange(len(input_labels_train)))[:self.batch_size]
                batch_input_ids = input_ids_train[shuffIndex]
                batch_input_mask = input_mask_train[shuffIndex]
                batch_segment_ids = segment_ids_train[shuffIndex]
                batch_input_labels = input_labels_train[shuffIndex]
                feed_dict = {
                           self.input_ids:batch_input_ids,
                           self.input_mask:batch_input_mask,
                           self.segment_ids:batch_segment_ids,
                           self.input_labels:batch_input_labels
                        }
                _, batch_loss, batch_acc = sess.run([self.train_op, self.loss, self.acc], feed_dict = feed_dict)
                print("\ttrain_acc : {}".format(batch_acc))
            saver.save(sess = sess, save_path = self.model_path)


    def eval(self, sess, input_ids_train, input_mask_train, segment_ids_train,
            input_labels_train):
        batches = batch_yield(input_ids_train, input_mask_train,
                              segment_ids_train, input_labels_train,
                              self.batch_size, self.is_shuffle)
        total_loss, total_acc = 0.0, 0.0
        for input_ids_batch, input_mask_batch, segment_ids_batch, input_labels_batch in batches:
            batch_len = len(input_labels_batch)
            feed_dict = {self.input_ids : input_ids_batch,
                         self.input_mask : input_mask_batch,
                         self.segment_ids : segment_ids_batch,
                         self.input_labels : input_labels_batch}
            loss, acc = sess.run([self.loss, self.acc], feed_dict = feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        total_len = len(x)
        return total_loss / total_len, total_acc / total_len

    def predict(self, input_ids, input_mask, segment_ids):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess = sess, save_path = self.model_path)
            data_size = len(input_ids)
            num_batches = int((data_size - 1) / self.batch_size) + 1
            y_pred = np.zeros(shape = data_size, dtype = np.int32)
            y_prob_mat = np.zeros(shape = (data_size, self.num_labels), dtype = np.float32)
            for i in range(num_batches):
                start_idx = i * (self.batch_size)
                end_idx = min((i + 1) * (self.batch_size), data_size)
                feed_dict = {self.input_ids : input_ids[start_idx : end_idx],
                             self.input_mask : input_mask[start_idx : end_idx],
                             self.segment_ids : segment_ids[start_idx : end_idx]}
                y_pred[start_idx : end_idx], y_prob_mat[start_idx : end_idx] = sess.run([self.prediction, self.softmax_probabilities], feed_dict = feed_dict)
                print("finish batch {}".format(i + 1))

        return y_pred, y_prob_mat


