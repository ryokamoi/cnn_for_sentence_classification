import os
import shutil

import numpy as np
import tensorflow as tf

from model import Model
from config import FLAGS
from batchloader import BatchLoader

def log_and_print(log_file, logstr, br=True):
    print(logstr)

    if(br):
        logstr = logstr + "\n"
    with open(log_file, 'a') as f:
        f.write(logstr)

def main():
    os.mkdir(FLAGS.LOG_DIR)
    os.mkdir(FLAGS.LOG_DIR + "/model")
    log_file = FLAGS.LOG_DIR + "/log.txt"
    shutil.copyfile("config.py", FLAGS.LOG_DIR + "/config.py")

    # gpu memory
    sess_conf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            # per_process_gpu_memory_fraction=0.4,
            # allow_growth = True
        )
    )

    with tf.Graph().as_default():
        with tf.Session(config=sess_conf) as sess:
            batchloader = BatchLoader()

            with tf.variable_scope("Model"):
                model_train = Model(batchloader,
                                    is_training=True)

            with tf.variable_scope("Model", reuse=True):
                model_val = Model(batchloader,
                                  is_training=False)

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.LOG_DIR, sess.graph)

            sess.run(tf.global_variables_initializer())

            log_and_print(log_file, "start training")

            loss_log = []
            accuracy_log = []
            lr = FLAGS.LEARNING_RATE
            step = 0
            for epoch in range(FLAGS.EPOCH):
                log_and_print(log_file, "epoch %d" % (epoch+1))
                if epoch >= FLAGS.LR_DECAY_START:
                    lr *= 0.95
                for batch in range(FLAGS.BATCHES_PER_EPOCH):

                    step += 1

                    input_text, label = batchloader.next_batch(FLAGS.BATCH_SIZE, "train")

                    feed_dict = {model_train.input_text: input_text,
                                 model_train.label: label,
                                 model_train.lr: lr}

                    loss, accuracy, merged_summary, _ \
                            = sess.run([model_train.loss, \
                                        model_train.accuracy, \
                                        model_train.merged_summary, \
                                        model_train.train_op],
                                        feed_dict = feed_dict)

                    loss_log.append(loss)
                    accuracy_log.append(accuracy)
                    summary_writer.add_summary(merged_summary, step)

                    # log
                    if(batch % 100 == 99):
                        log_and_print(log_file, "epoch %d batch %d" % \
                                                ((epoch+1), (batch+1)), br=False)

                        ave_loss = np.average(loss_log)
                        log_and_print(log_file, "\ttrain loss: %f" % ave_loss, br=False)
                        ave_acc = np.average(accuracy_log)
                        log_and_print(log_file, "\ttrain accuracy: %f" % ave_acc, br=False)

                        loss_log = []
                        accuracy_log = []

                        # valid output
                        input_text, label = batchloader.next_batch(FLAGS.BATCH_SIZE, "valid")

                        feed_dict = {model_val.input_text: input_text,
                                     model_val.label: label}

                        loss, accuracy, merged_summary \
                                = sess.run([model_val.loss, \
                                            model_val.accuracy, \
                                            model_val.merged_summary],
                                            feed_dict = feed_dict)

                        log_and_print(log_file, "\tval loss: %f" % loss, br=False)
                        log_and_print(log_file, "\tval accuracy: %f" % accuracy)

                        summary_writer.add_summary(merged_summary, step)

                # save model
                save_path = saver.save(sess, FLAGS.LOG_DIR + ("/model/model%d.ckpt" % (epoch+1)))
                log_and_print(log_file, "Model saved in file %s" % save_path)


if __name__ == "__main__":
    main()
