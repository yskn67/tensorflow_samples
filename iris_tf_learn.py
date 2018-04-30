#! /usr/bin/env python

import numpy as np
import tensorflow as tf
import urllib.request


IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

with open(IRIS_TRAINING, 'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TRAINING_URL).read())

with open(IRIS_TEST, 'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TEST_URL).read())

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir='./iris_model')


def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y


def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y


classifier.fit(input_fn=get_train_inputs, steps=2000)
print(classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"])
