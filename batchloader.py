import pickle as pkl

import numpy as np

from config import FLAGS


class BatchLoader:
    def __init__(self):
        self.go_token = '<GO>'
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        with open(FLAGS.TRAIN_PATH, "rb") as f:
            self.train_data = pkl.load(f)

        with open(FLAGS.TRAIN_LABEL_PATH, "rb") as f:
            self.train_label = pkl.load(f)

        if FLAGS.DATASET_NUM != -1:
            self.train_data = self.train_data[:FLAGS.DATASET_NUM]
            self.train_label = self.train_label[:FLAGS.DATASET_NUM]

        with open(FLAGS.VALID_PATH, "rb") as f:
            self.valid_data = pkl.load(f)

        with open(FLAGS.VALID_LABEL_PATH, "rb") as f:
            self.valid_label = pkl.load(f)

        with open(FLAGS.TEST_PATH, "rb") as f:
            self.test_data = pkl.load(f)

        with open(FLAGS.TEST_LABEL_PATH, "rb") as f:
            self.test_label = pkl.load(f)

        with open(FLAGS.DICT_PATH, "rb") as f:
            self.char_to_idx = pkl.load(f)

        self.idx_to_char = {}
        for char, idx in self.char_to_idx.items():
            self.idx_to_char[idx] = char


    def next_batch(self, batch_size, target: str):
        if target == "train":
            indexes = np.array(np.random.randint(len(self.train_data), size=batch_size))
            text = np.array([np.copy(self.train_data[idx]).tolist() for idx in indexes])

            label = np.array([self.train_label[idx] for idx in indexes])

            return text, label

        else:
            indexes = np.array(np.random.randint(len(self.valid_data), size=batch_size))
            text = np.array([np.copy(self.valid_data[idx]).tolist() for idx in indexes])

            label = np.array([np.copy(self.valid_label[idx]).tolist() for idx in indexes])

            return text, label
