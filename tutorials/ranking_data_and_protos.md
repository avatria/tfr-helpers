# Introduction to TensorFlow features and protocol buffers

If you've noticed, TensorFlow models don't exactly follow the scikit-learn API you may be used to as a python data guru. Since there are no simple `.fit`, `.transform`, and `.predict` methods on a TensorFlow `Estimator`, you're 

### Background

TensorFlow, originally released at the end of 2015, has been actively evolving owing to the dedication of thousands of active contributors. Its main competitor, as of 2020, seems to be PyTorch. One of its prior competitors, Keras, mainly provided wrappers around the TensorFlow graph API and exposed a much more scikit-learn-like interface to users for building models. For example Keras offers a `Model.fit` method, resembling that of a traditional scikit-learn estimator. By mid 2019, Tensorflow was on version 1.0.

By this point, Keras was growing in popularity because it was so simple to use relative to version 1 TensorFlow code. Since Keras was developed by a Google engineer and its main dependency was TensorFlow itself, Keras and Tensorflow were officially married in TensorFlow v2 in late 2019. Most of the v1 functionality is retained in v2 within the `compat.v1` submodule (`from tensorflow.compat.v1 import *`) should you need it.

Well, `tensorflow_ranking` does need it, so you'll frequently see v1 functionality referenced within TensorFlow Ranking code and examples while also seeing Tensorflow Ranking models implemented as Keras models. In fact, there is an entire `tensorflow_ranking.keras` submodule where Keras versions of ranking estimators have been implemented:

```
import tensorflow_ranking as tfr

model = tfr.keras.canned.DNNRankingNetwork(...)
```

Still, even the Keras versions of TensorFlow Ranking estimators remain rather cryptic relative to a typical Keras model. We'll see in later tutorials exactly what all of these arguments that you may have stumbled upon like `context_feature_columns` and `example_feature_columns` mean.

### Ranking data formats

Ranking is not as simple as classification and regression machine learning tasks. Those types of models generally assume *iid* datasets, i.e. sample *i* is independent of sample *j*. The same is not true in ranking datasets.

