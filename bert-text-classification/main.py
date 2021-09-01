# coding:utf-8

import io
import os
import data_loader
import numpy as np
import tensorflow as tf
from bert import modeling
from model import BertTextClassification

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string("bert_config_file", None, 
        "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("data_dir", None, "The input data dir.")

flags.DEFINE_string("init_checkpoint", None, 
        "Initial checkpoint from bert-base-chinese model.")

flags.DEFINE_string("vocab_file", None, 
        "The vocabulary file that the BERT model was trained on.")

# Other parameters
flags.DEFINE_integer("max_seq_length", 128, 
    "The maximum total input sequence length after character tokenization.")

flags.DEFINE_integer("max_epoches", 100, "Maximum epoch number.")

flags.DEFINE_integer("num_labels", 2, "Number of classification labels.")

flags.DEFINE_integer("batch_size", 20, "Number of minibatch samples.")

flags.DEFINE_bool("is_training", True, "Whether to run training.")

flags.DEFINE_float("learning_rate", 0.00005, "The initial learning rate for optimizer.")

flags.DEFINE_bool("is_shuffle", True, "Whether to shuffle data for generating batches.")

flags.DEFINE_string("optimizer", "Adam", "The optimizer for training.")

flags.DEFINE_float("clip_grad", 5.0, "Gradient clipping.")

flags.DEFINE_bool("use_one_hot_embeddings", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the test set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the single text.")

def save_predictions(output_file, y_pred, y_prob_mat):
    f = io.open(output_file, mode = 'w', encoding = 'utf-8')
    f.write(("label" + "\t" + "probability" + "\n").decode('unicode_escape'))
    for i in range(len(y_pred)):
        f.write(str(y_pred[i]).decode('unicode_escape'))
        for j in range(y_prob_mat.shape[1]):
            f.write(("\t" + str(y_prob_mat[i, j])).decode('unicode_escape'))
        f.write(("\n").decode('unicode_escape'))
    f.close()


def main(_):
    paths = {}
    dir = os.path.join(".", "result")
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_prefix = os.path.join(dir, "latest_model")
    log_path = os.path.join(dir, "log.txt")
    result_path = os.path.join(dir, "prediction.txt")
    paths["model_path"] = model_prefix
    paths["log_path"] = log_path
    paths["result_path"] = result_path

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" % (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    model_fn = BertTextClassification(bert_config, FLAGS.batch_size,
            FLAGS.max_seq_length, FLAGS.is_shuffle, FLAGS.optimizer,
            FLAGS.learning_rate, FLAGS.max_epoches, FLAGS.init_checkpoint,
            FLAGS.clip_grad, FLAGS.num_labels, FLAGS.is_training,
            FLAGS.use_one_hot_embeddings, paths)
    model_fn.build()

    if FLAGS.do_train:
        input_file = os.path.join(FLAGS.data_dir, "input_text.txt")
        input_ids, input_mask, segment_ids, labels = data_loader.read_corpus(input_file, FLAGS.vocab_file, FLAGS.max_seq_length)
        model_fn.train(input_ids, input_mask, segment_ids, labels)
    if FLAGS.do_eval:
        predict_file = os.path.join(FLAGS.data_dir, "predict_text.txt")
        input_ids, input_mask, segment_ids, _ = data_loader.read_corpus(predict_file, FLAGS.vocab_file, FLAGS.max_seq_length)
        y_pred, y_prob_mat = model_fn.predict(input_ids, input_mask, segment_ids)
        save_predictions(paths["result_path"], y_pred, y_prob_mat)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    tf.app.run()

