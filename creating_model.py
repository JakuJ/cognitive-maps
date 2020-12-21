import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
import pandas as pd
import os
import itertools
import skfuzzy as fuzz
from sklearn.preprocessing import minmax_scale
import argparse
import sys
import csv
import losses as L

WINDOW_SIZE = 0
CONCEPTS = 0
PATH = ""


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
    t = 5
    return 1.0 / (1.0 + K.exp(-t * x))


def make_model(input_size, output_size, window, use_constraint=True, use_bias=True):
    constraint = Between(-1, 1) if use_constraint else None

    inputs = [layers.Input(shape=(window,), name=f"concept_{i}") for i in range(input_size)]
    denses = [layers.Dense(1, activation, use_bias=use_bias, kernel_constraint=constraint, name=f'aggregate_{i}')(inp) for i, inp in enumerate(inputs)]

    if input_size > 1:
        aggregation = layers.concatenate(denses)
    else:
        aggregation = denses[0]

    fcm = layers.Dense(output_size, activation, use_bias=use_bias, kernel_constraint=constraint, name='fcm')(aggregation)
    return keras.Model(inputs=inputs, outputs=[fcm])


def window_gen(X, k):
    n = X.shape[0]
    for i in range(n - k + 1):
        window = X[i:i+k, :]
        yield [tuple(x) for x in window[:]]


def windows(X, k):
    return np.array(list(window_gen(X, k)))


def load_file(folder, category, f):
    df = pd.read_csv(f'{PATH}/{folder}/{category}/{f}', header=None)
    normalized = minmax_scale(df.values)
    df = pd.DataFrame(normalized)
    df.columns = ['x', 'y', 'z']
    return df


def to_6D_space(X):
    dX = X[1:, :] - X[:-1, :]
    X = np.hstack([X[1:, :], dX])
    return minmax_scale(X)


def to_concepts(X, centers):
    X_fuzzy, *_ = fuzz.cluster.cmeans_predict(X.T, centers, 2, error=0.005, maxiter=1000, init=None)
    return X_fuzzy.T


def windows_to_inputs(X_windows):
    """Windows to Xs, Ys tuple"""
    return {f"concept_{i}": X_windows[:-1, :, i] for i in range(CONCEPTS)}


def load_category(folder, category):
    path = f'{PATH}/{folder}/{category}'
    files = os.listdir(path)

    # find fuzzy centroids for a single category
    XdXs = []
    for f in files:
        # load 3D data
        df = load_file(folder, category, f)
        # to 6D (X, Y, Z, dX, dY, dZ) space
        XdX = to_6D_space(df.values)
        XdXs.append(XdX)

    # to np.array: files x length (314) x features (6)
    data = np.array(XdXs)
    return data


def find_centroids(XdXs):
    # concatenate ALL 6D data in category
    X = np.vstack(XdXs) # N x 6
    centers, *_ = fuzz.cluster.cmeans(X.T, CONCEPTS, 2, error=0.005, maxiter=10000, init=None)
    return centers


def data_to_generator(data, centroids):
    for XdX in data:
        # to fuzzy concept-space
        X_fuzzy = to_concepts(XdX, centroids)

        # split into windows of size WINDOW_SIZE
        X_windows = windows(X_fuzzy, WINDOW_SIZE)

        # split into separate inputs for the model
        yield windows_to_inputs(X_windows), X_windows[1:, -1, :]


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Parses commands")
    parser.add_argument("-w", "--window_size", type=int, help="Window size")
    parser.add_argument("-f", "--features", type=int, help="Number of the features")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-c", "--category", help="Name of the category")
    parser.add_argument("-s", "--source", help="Path to UWaveGestureLibrary")
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
    options = parser.parse_args(args)
    return options


if __name__ == "__main__":
    options = getOptions(sys.argv[1:])

    if not os.path.isdir(options.checkpoint_path):
        raise Exception("The folder doesn't exist")

    WINDOW_SIZE = options.window_size
    CONCEPTS = options.concepts
    PATH = options.source

    dataTrain = load_category('Train', options.category)
    dataTest = load_category('Test', options.category)
    centroids = find_centroids(dataTrain)

    np.savetxt(f"{options.checkpoint_path}/centroids.csv", centroids, delimiter=", ", fmt='% s')
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
        error = L.mean_absolute_relative_error
    elif options.loss == 6:
        error = L.mean_squared_relative_error
    elif options.loss == 7:
        error = L.smae_loss

    optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer, error)
    train_data = itertools.cycle(data_to_generator(dataTrain, centroids))
    validation_data = itertools.cycle(data_to_generator(dataTest, centroids))

    history = model.fit(train_data, epochs=options.epochs, validation_data=validation_data, validation_steps=50, steps_per_epoch=500, verbose=1)
    model.save(f'{options.checkpoint_path}')
