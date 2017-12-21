from datetime import datetime

import tensorflow as tf

flags = tf.app.flags

data_dir = "dataset/"
flags.DEFINE_string('TRAIN_PATH', data_dir + "train.pkl", "")
flags.DEFINE_string('TRAIN_LABEL_PATH', data_dir + "train_label.pkl", "")

flags.DEFINE_string('VALID_PATH', data_dir + "val.pkl", "")
flags.DEFINE_string('VALID_LABEL_PATH', data_dir + "val_label.pkl", "")

flags.DEFINE_string('TEST_PATH', data_dir + "test.pkl", "")
flags.DEFINE_string('TEST_LABEL_PATH', data_dir + "test_label.pkl", "")

flags.DEFINE_string('DICT_PATH', data_dir + "dictionary.pkl", "")

flags.DEFINE_integer('VOCAB_SIZE', 10000, '')
flags.DEFINE_integer('BATCH_SIZE', 64, '')
flags.DEFINE_integer('SEQ_LEN', 30, '')
flags.DEFINE_integer('CLASS_NUM', 2, '')

flags.DEFINE_integer('DATASET_NUM', -1, '') # set -1 to use all data

flags.DEFINE_integer('EPOCH', 10, '')
flags.DEFINE_integer('BATCHES_PER_EPOCH', 5000, '')

flags.DEFINE_integer('LEARNING_RATE', 0.01, '')
flags.DEFINE_integer('DROPOUT_KEEP', 0.5, '')
flags.DEFINE_integer('LR_DECAY_START', 5, '')
flags.DEFINE_integer('MAX_GRAD', 5.0, '')

flags.DEFINE_integer('EMBED_SIZE', 300, '')
flags.DEFINE_integer('FILTER_SIZE', [3, 4, 5], '')
flags.DEFINE_integer('FILTER_NUM', 32, '')

flags.DEFINE_string('LOG_DIR', "log/log" + datetime.now().strftime("%y%m%d-%H%M"), "")

FLAGS = flags.FLAGS
