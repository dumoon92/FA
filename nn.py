import pathlib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import scipy.io
# data = scipy.io.loadmat('088IRWaSS7_Wi1d89_C4d3_wave.mat')['WG10_DHI']
data = np.transpose(scipy.io.loadmat('matlab.mat')['data'])


def norm(x):
    return (x - np.mean(x)) / np.std(x)


data = norm(data)

train_start = 0
train_len = 100
train_set = 100

predict_len = 10

train_x = np.zeros((train_set, train_len))
train_y = np.zeros((train_set, predict_len))
for i in range(train_set):
    train_x[i, :] = data[0, i: i+train_len]
    train_y[i, :] = data[0, i+train_len: i+train_len+predict_len]


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()
model.summary()

example_batch = train_x[:10]
example_result = model.predict(example_batch)
example_result

