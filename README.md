# TensorFlow implementation of "Convolutional Neural Networks for Sentence Classification"

paper:[https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)

This is NOT an original implementation. There may be some minor differences from the original structure.

## Prerequisites

 * Python 3.5
 * tensorflow-gpu==1.3.0
 * numpy==1.13.1
 * jupyter==1.0.0

## TO-DO

 * Use pre-trained vectores from word2vec
     * Currently, word embeddings are randomly initialized

## Preparation
### Download Dataset

1. Download "sentence polarity dataset v1.0" from [http://www.cs.cornell.edu/people/pabo/movie-review-data/](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
2. run "data_prepare.ipynb" with jupyter notebook

## Usage
### Train

1. modify "config.py"
2. run

```bash
  python train.py
```

### Get test result

1. modify sampling.py
2. run

```bash
  python test.py
```

## License

MIT

## Author

Ryo Kamoi
