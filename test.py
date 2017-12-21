import pickle as pkl

import numpy as np
import tensorflow as tf

from train import log_and_print
from model import Model
from config import FLAGS
from batchloader import BatchLoader

LOG_DIR = "log/log171221-2343"
MODEL_DIR = LOG_DIR + "/model"
SAVE_FILE = LOG_DIR + "/test.txt"

def test():
    batchloader = BatchLoader()

    # gpu memory
    sess_conf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            # per_process_gpu_memory_fraction=0.4,
            # allow_growth = True
        )
    )

    with tf.Graph().as_default():
        with tf.Session(config=sess_conf) as sess:
            with tf.variable_scope("Model"):
                model_restored = Model(batchloader, is_training=False)

            saver = tf.train.Saver()
            saver.restore(sess, MODEL_DIR + "/model10.ckpt")

            with open(FLAGS.TEST_PATH, "rb") as f:
                test_data = pkl.load(f)

            sample_num = len(test_data)
            log_and_print(SAVE_FILE, "sample_num: %d" % (FLAGS.BATCH_SIZE * (sample_num//FLAGS.BATCH_SIZE)))

            with open(FLAGS.TEST_LABEL_PATH, "rb") as f:
                test_label = pkl.load(f)

            accuracy_save = []
            for i in range(sample_num//FLAGS.BATCH_SIZE):
                tmp_data = test_data[FLAGS.BATCH_SIZE*i:(FLAGS.BATCH_SIZE*(i+1))]
                tmp_label = test_label[FLAGS.BATCH_SIZE*i:FLAGS.BATCH_SIZE*(i+1)]

                accuracy = sess.run(model_restored.accuracy,
                                    feed_dict={model_restored.input_text: tmp_data,
                                               model_restored.label: tmp_label})

                accuracy_save.append(accuracy)

            log_and_print(SAVE_FILE, "accuracy: %f" % np.average(accuracy_save))


if __name__ == "__main__":
    test()
