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

training_data = np.loadtxt(IRIS_TRAINING, delimiter=',', skiprows=1)
train_x = training_data[:, :-1]
train_y = training_data[:, -1]

test_data = np.loadtxt(IRIS_TEST, delimiter=',', skiprows=1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]


keras = tf.keras
callbacks = tf.keras.callbacks
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
backend = keras.backend

num_classes = 3
train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

tensorboard = callbacks.TensorBoard(log_dir='./logs/', histogram_freq=1)
callback_list = [tensorboard]

history = model.fit(train_x, train_y,
                    batch_size=20,
                    epochs=2000,
                    verbose=0,
                    validation_data=(test_x, test_y),
                    callbacks=callback_list)

score = model.evaluate(test_x, test_y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

backend.clear_session()
