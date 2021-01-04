import argparse
import glob
import itertools
import os
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint

import losses as L
from common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Between(Constraint):
    """Kernel constraint keeping model weights between the values provided at initialization."""

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


def make_model(input_size, output_size, window_width):
    constraint = Between(-1, 1)

    input_layers = [layers.Input(shape=(window_width,), name=f"concept_{i}") for i in range(input_size)]
    aggregation_layers = [layers.Dense(1, activation, kernel_constraint=constraint, name=f'aggregate_{i}')(inp) for i, inp in enumerate(input_layers)]

    if input_size > 1:
        aggregation = layers.concatenate(aggregation_layers)
    else:
        aggregation = aggregation_layers[0]

    fcm = layers.Dense(output_size, activation, kernel_constraint=constraint, name='fcm')(aggregation)
    return keras.Model(inputs=input_layers, outputs=[fcm])


def load_folder(path):
    files = os.listdir(path)

    # load all CSV files and transform to 2N space (data + derivatives)
    XdXs = [expand_derivatives(load_file(os.path.join(path, f))) for f in files]

    # dimensions: files x length x features
    return np.array(XdXs)


def find_centroids(XdXs, num_concepts):
    # concatenate all data points
    X = np.vstack(XdXs)

    # find fuzzy cluster centers
    centers, *_ = fuzz.cluster.cmeans(X.T, num_concepts, 2, error=0.005, maxiter=10000, init=None)

    return centers


def data_to_generator(data, centers, num_concepts, window_width):
    for XdX in data:
        # to fuzzy concept-space
        X_fuzzy = to_concepts(XdX, centers)

        # split into windows
        X_windows = windows(X_fuzzy, window_width, skip_last=False)

        # split into separate inputs and outputs for the model
        yield windows_to_inputs(X_windows, num_concepts, skip_last=True), X_windows[1:, -1, :]


def getOptions(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--train-source", help="path to training data folder", required=True)
    parser.add_argument("--test-source", help="path to test data folder", required=True)
    parser.add_argument("--model-path", help="path to the folder with model data", required=True)
    parser.add_argument("-c", "--concepts", type=int, help="number of concepts", required=True)
    parser.add_argument("-w", "--window-size", type=int, help="window size, defaults to 30", default=30)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs, defaults to 10", default=10)
    parser.add_argument("-r", "--learning-rate", type=float, help="learning rate, defaults to 0.1", default=0.1)
    parser.add_argument("-l", "--loss", type=int, help='''loss function to be used for training. 
1 - Mean squared error (default)
2 - Mean absolute error
3 - Mean squared logarithmic error
4 - Mean relative error
5 - Mean squared relative error
6 - Symmetric mean absolute error''', default=1)
    return parser.parse_args(args)


if __name__ == "__main__":
    # parse options
    options = getOptions(sys.argv[1:])

    if not len(glob.glob(f"{options.train_source}/*.csv")):
        raise Exception(f"{options.train_source} - training data folder contains no CSV files")

    if not len(glob.glob(f"{options.test_source}/*.csv")):
        raise Exception(f"{options.test_source} - test data folder contains no CSV files")

    # load data
    dataTrain = load_folder(options.train_source)
    dataTest = load_folder(options.test_source)
    centroids = find_centroids(dataTrain, options.concepts)

    # choose error function
    error = [
        L.mean_squared_error,
        L.mean_absolute_error,
        L.mean_squared_logarithmic_error,
        tf.function(L.mean_relative_error),
        tf.function(L.mean_squared_relative_error),
        tf.function(L.symmetric_mean_absolute_error),
    ][options.loss - 1]

    # create model
    model = make_model(options.concepts, options.concepts, options.window_size)
    optimizer = keras.optimizers.SGD(learning_rate=options.learning_rate)
    model.compile(optimizer, error)

    # prepare data
    train_data = itertools.cycle(data_to_generator(dataTrain, centroids, options.concepts, options.window_size))
    validation_data = itertools.cycle(data_to_generator(dataTest, centroids, options.concepts, options.window_size))

    # train model
    history = model.fit(train_data, epochs=options.epochs, validation_data=validation_data, validation_steps=50, steps_per_epoch=500, verbose=1)

    # serialize model and centroids
    model.save(options.model_path)
    centroids_path = os.path.join(options.model_path, 'centroids.csv')
    np.savetxt(centroids_path, centroids, delimiter=", ", fmt='%s')

    print("Done")
