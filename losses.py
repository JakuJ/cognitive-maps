"""This module contains loss functions compatible with the Keras API."""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

mean_squared_error = keras.losses.mean_squared_error
"""Mean squared error loss function."""

mean_absolute_error = keras.losses.mean_absolute_error
"""Mean absolute error loss function."""

mean_squared_logarithmic_error = keras.losses.mean_squared_logarithmic_error
"""Mean squared logarithmic error loss function."""


def huber_loss(x, y):
    """Huber loss function with delta=0.25."""
    return keras.losses.huber(x, y, 0.25)


@tf.function
def mean_absolute_relative_error(y_true, y_pred):
    """Mean absolute relative error function."""
    return K.mean(K.abs((y_true - y_pred) / y_true), axis=-1)


@tf.function
def mean_squared_relative_error(y_true, y_pred):
    """Mean squared relative error function."""
    return K.mean(K.square(y_true - y_pred) / y_true, axis=-1)


@tf.function
def smae_loss(y_true, y_pred):
    """Symmetric mean absolute error function."""
    return K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)), axis=-1)


if __name__ == "__main__":
    import sys
    print(sys.stderr, "This module shouldn't be ran on its own")
