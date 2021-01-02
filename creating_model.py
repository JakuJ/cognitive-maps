import argparse
import glob
import itertools
import os
import sys

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint

import losses as L
from common import *

WINDOW_SIZE = 0
CONCEPTS = 0


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {
            'min_value': self.min_value,
            'max_value': self.max_value
        }


@tf.function
def activation(x):
    """Sigmoid function with theta = 5"""
    return 1.0 / (1.0 + K.exp(-5 * x))


def make_model(input_size, output_size, window):
    constraint = Between(-1, 1)

    input_layers = [layers.Input(shape=(window,), name=f"concept_{i}") for i in range(input_size)]
    aggregation_layers = [layers.Dense(1, activation, kernel_constraint=constraint, name=f'aggregate_{i}')(inp) for i, inp in enumerate(input_layers)]

    if input_size > 1:
        aggregation = layers.concatenate(aggregation_layers)
    else:
        aggregation = aggregation_layers[0]

    fcm = layers.Dense(output_size, activation, kernel_constraint=constraint, name='fcm')(aggregation)
    return keras.Model(inputs=input_layers, outputs=[fcm])


def window_generator(X, k):
    n = X.shape[0]
    for i in range(n - k + 1):
        window = X[i:i + k, :]
        yield [tuple(x) for x in window[:]]


def windows(X, k):
    return np.array(list(window_generator(X, k)))


def load_file(path, f):
    df = pd.read_csv(f'{path}/{f}', header=None)
    return minmax_scale(df.values)


def windows_to_inputs(X_windows):
    """Windows to Xs, Ys tuple"""
    return {f"concept_{i}": X_windows[:-1, :, i] for i in range(CONCEPTS)}


def load_folder(path):
    files = os.listdir(path)

    # load all CSV files and transform to 2N space (data + derivatives)
    XdXs = [to_6D_space(load_file(path, f)) for f in files]

    # to np.array: files x length x features
    return np.array(XdXs)


def find_centroids(XdXs):
    # concatenate all data points
    X = np.vstack(XdXs)

    # find fuzzy cluster centers
    centers, *_ = fuzz.cluster.cmeans(X.T, CONCEPTS, 2, error=0.005, maxiter=10000, init=None)

    return centers


def data_to_generator(data, centers):
    for XdX in data:
        # to fuzzy concept-space
        X_fuzzy = to_concepts(XdX, centers)

        # split into windows
        X_windows = windows(X_fuzzy, WINDOW_SIZE)

        # split into separate inputs and outputs for the model
        yield windows_to_inputs(X_windows), X_windows[1:, -1, :]


def getOptions(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Parses commands")
    parser.add_argument("-w", "--window_size", type=int, help="Window size")
    parser.add_argument("-f", "--features", type=int, help="Number of the features")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("--train_source", help="Path to Train Set")
    parser.add_argument("--test_source", help="Path to Test Set")
    parser.add_argument("-n", "--concepts", type=int, help="Number of concepts")
    parser.add_argument("-l", "--loss", type=int, help='''Loss function to be used for training. 
    1 - Mean Squared Error 
    2 - Mean Absolute Error
    3 - Mean Squared Logarithmic Error
    4 - Huber Loss
    5 - Mean Absolute Percentage Error
    6 - Mean Squared Percentage Error
    7 - Symmetric Mean Absolute Percentage Error''')
    parser.add_argument("--checkpoint_path", help="Path to the folder with checkpoint data")
    return parser.parse_args(args)


if __name__ == "__main__":
    options = getOptions(sys.argv[1:])

    if not os.path.isdir(options.checkpoint_path):
        raise Exception(f"{options.checkpoint_path} - folder doesn't exist")

    if not len(glob.glob(f"{options.train_source}/*.csv")):
        raise Exception(f"{options.train_source} - training data folder contains no CSV files")

    if not len(glob.glob(f"{options.test_source}/*.csv")):
        raise Exception(f"{options.test_source} - test data folder contains no CSV files")

    WINDOW_SIZE = options.window_size
    CONCEPTS = options.concepts

    dataTrain = load_folder(options.train_source)
    dataTest = load_folder(options.test_source)
    centroids = find_centroids(dataTrain)

    centroids_path = os.path.join(options.checkpoint_path, 'centroids.csv')
    np.savetxt(centroids_path, centroids, delimiter=", ", fmt='%s')
    model = make_model(CONCEPTS, CONCEPTS, WINDOW_SIZE)

    if options.loss == 1:
        error = L.mean_squared_error
    elif options.loss == 2:
        error = L.mean_absolute_error
    elif options.loss == 3:
        error = L.mean_squared_logarithmic_error
    elif options.loss == 4:
        error = L.huber_loss
    elif options.loss == 5:
        error = L.mean_absolute_relative_error_tf
    elif options.loss == 6:
        error = L.mean_squared_relative_error_tf
    elif options.loss == 7:
        error = L.symetric_mean_absolute_error_tf

    optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer, error)
    train_data = itertools.cycle(data_to_generator(dataTrain, centroids))
    validation_data = itertools.cycle(data_to_generator(dataTest, centroids))

    history = model.fit(train_data, epochs=options.epochs, validation_data=validation_data, validation_steps=50, steps_per_epoch=500, verbose=1)
    model.save(options.checkpoint_path)
