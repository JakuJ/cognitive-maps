import tensorflow.keras.backend as K
from tensorflow import keras

mean_squared_error = keras.losses.mean_squared_error
mean_absolute_error = keras.losses.mean_absolute_error
mean_squared_logarithmic_error = keras.losses.mean_squared_logarithmic_error


def mean_relative_error(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / y_true), axis=-1)


def max_relative_error(y_true, y_pred):
    return K.max(K.abs((y_true - y_pred) / y_true), axis=-1)


def mean_squared_relative_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred) / y_true, axis=-1)


def max_squared_relative_error(y_true, y_pred):
    return K.max(K.square(y_true - y_pred) / y_true, axis=-1)


def symmetric_mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)), axis=-1)
