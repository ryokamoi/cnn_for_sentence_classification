import tensorflow as tf

from config import FLAGS

class Model(object):
    def __init__(self, batchloader, is_training=True):
        self.batchloader = batchloader
        self.is_training = is_training
        self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        with tf.name_scope("Placeholders"):
            self.input_text = tf.placeholder(tf.int32,
                                             shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                             name="input_text")

            self.label = tf.placeholder(tf.int32,
                                        shape=(FLAGS.BATCH_SIZE),
                                        name="label")


        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable(name="embedding",
                                 shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))


        with tf.name_scope("Embedding_text"):
            # NHWC
            self.embedded_text = tf.expand_dims(tf.nn.embedding_lookup(self.embedding, self.input_text),
                                                axis=-1)
            assert self.embedded_text.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.EMBED_SIZE, 1)


        with tf.variable_scope("CNN"):
            self.maxpooled_output = []
            self.cnn_W = []
            for i, filter_size in enumerate(FLAGS.FILTER_SIZE):
                layer_name = "cnn_layer_%d_%d" % (i, filter_size)
                with tf.variable_scope(layer_name):
                    self.cnn_W.append(tf.get_variable(name=layer_name+"_W",
                                                      shape=(filter_size,
                                                             FLAGS.EMBED_SIZE,
                                                             1,
                                                             FLAGS.FILTER_NUM),
                                                      dtype=tf.float32,
                                                      initializer=tf.truncated_normal_initializer(stddev=0.1)))

                    conv = tf.nn.conv2d(self.embedded_text,
                                        self.cnn_W[i],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name=layer_name+"_conv")

                    maxpooled = tf.nn.max_pool(conv,
                                               ksize=[1, FLAGS.SEQ_LEN - filter_size + 1, 1, 1],
                                               strides=[1, 1, 1, 1],
                                               padding="VALID")

                    self.maxpooled_output.append(maxpooled)


            with tf.variable_scope("Feaure"):
                self.feature = tf.reshape(tf.concat(self.maxpooled_output, axis=3),
                                          shape=[FLAGS.BATCH_SIZE, FLAGS.FILTER_NUM * len(FLAGS.FILTER_SIZE)])

                if self.is_training:
                    self.feature = tf.nn.dropout(self.feature, FLAGS.DROPOUT_KEEP)


        with tf.variable_scope("Linear"):
            self.W = tf.get_variable(name="W",
                                     shape=(FLAGS.FILTER_NUM * len(FLAGS.FILTER_SIZE),
                                            FLAGS.CLASS_NUM),
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.b = tf.get_variable(name="b",
                                     shape=(FLAGS.CLASS_NUM),
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())


        with tf.name_scope("Logits"):
            self.logits = tf.matmul(self.feature, self.W) + self.b


        with tf.name_scope("Accuracy"):
            prediction = tf.argmax(self.logits, 1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(
                                tf.cast(tf.equal(prediction, self.label),
                                        dtype=tf.float32))


        with tf.name_scope("Loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.label,
                        logits=self.logits,
                        name="loss")
            self.loss = tf.reduce_mean(losses)


        with tf.name_scope("Summary"):
            if is_training:
                loss_summary = tf.summary.scalar("loss", self.loss, family="train_loss")
                acc_summary = tf.summary.scalar("accuracy", self.accuracy, family="train_loss")
                lr_summary = tf.summary.scalar("lr", self.lr, family="parameters")

                self.merged_summary = tf.summary.merge([loss_summary, lr_summary, acc_summary])

            else:
                loss_summary = tf.summary.scalar("valid_loss", self.loss, family="val_loss")
                acc_summary = tf.summary.scalar("valid_accuracy", self.accuracy, family="val_loss")

                self.merged_summary = tf.summary.merge([loss_summary, acc_summary])


        if self.is_training:
            tvars = tf.trainable_variables()
            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars))
