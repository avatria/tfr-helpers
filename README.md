# tfr-helpers

Guides to using TensorFlow Ranking, and functions for working with TFR in the wild.

Have you wanted to use [TensorFlow Ranking](https://github.com/tensorflow/ranking) for your ranking machine learning tasks? At [Avatria](https://www.avatria.com/), learning-to-rank is at the core of what we do, so the answer was an obvious "yes" for us. But nobody on the team had extensive TensorFlow experience, and we soon found that there was a lot of activation energy required to get off the ground and to the races with `tensorflow-ranking`. After our data science team got TFR working the way we wanted it, we wanted to share our learnings with you so that others can have an easier go of it.

There are two main components of this repo: a set of tutorials and guides that explain how and why TFR works the way it does. These guides can help you do everything from convert your .csv's to Google protobufs to setting a learning rate decay schedule. Second, a python package with utility methods for data format conversion, feature scaling, and model exporting.


## TFR tutorials

0. Introduction to Tensorflow features, Protocol Buffers (protobufs), and Example Lists with Context (ELWCs).
1. Getting your data to work with TensorFlow datasets for ranking.
2. Groupwise Scoring Functions and standing up the **key components** of the Tensorflow Ranking Estimator.
3. Exporting and importing `SavedModels`.
4. Performance hacks.

## Other TFR resources
Here are some links to learning resources that we found helpful for figuring out to use TensorFlow Ranking

- TensorFlow Ranking [Colab notebook](https://colab.research.google.com/github/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb) tutorial
- [Example LTR implementation](https://quantdare.com/learning-to-rank-with-tensorflow/) using the deprecated `tfr.data.libsvm_generator` data processor method.
- [Airbnb paper](https://arxiv.org/pdf/1810.09591.pdf) on their experiences implementing deep nets for ranking.
- [Master's thesis](https://prof.beuth-hochschule.de/fileadmin/prof/aloeser/shuaib_thesis.pdf) on learning to rank with TFR example
- [SVMLight documentation](http://svmlight.joachims.org/), a popular LTR data format
- Crucial [TFR GitHub issue](https://github.com/tensorflow/ranking/issues/196) on converting Python data structures to ELWCs.
- [Groupwise Scoring Functions paper](https://arxiv.org/abs/1811.04415) describes the core implementation TFR models.


## Code
The `tfr_helpers` package provides utilities for converting your ranking datasets and pandas DataFrames into ELWCs, a rather complex data structure.

Other functionality:
- Exporting ranker estimators to SavedModel protobuf format.
- Extract model predictions on new `.tfrecords` files.
- Load `.tfrecords` files containing ELWC protos into dataframes for inspection.


## Sample data

