import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
import sys
import pandas as pd
import matplotlib.pyplot as plt

ARGS_COUNT = 4
window_size = 30
debug_args = [
    "scriptName", 
    "drive/MyDrive/model", 
    "dataset/UWaveGestureLibrary/Test/1/1001.csv", 
    "OUT.csv"
]
args = debug_args if "--debug" in sys.argv else sys.argv


def usage():
    print("USAGE")
    print(f"./{args[0]} [path_to_model] [path_to_data] [output_path]")
    print(f"Example: ./{args[0]} /my/saved/model/dir /dataset/category1/data.csv /output/data_out.csv")


def windows(X, k):
    n = X.shape[1]
    for i in range(n - k):  # skipping last window
        window = X[:, i:i+k]
        yield [tuple(x) for x in window[:]]


def load_data_from_path(path):
    # load dataframe
    df = pd.read_csv(path, header=None)

    # to numpy array
    X = np.array(df).T

    # normalize to [0, 1]
    X = 0.5 * (1.0 + X / np.linalg.norm(X, axis=0))

    # split into windows of size window_size
    X_windows = np.array(list(windows(X, window_size)))

    # split into separate inputs for the model
    Xs = {
        "concept_0": X_windows[:-1, 0, :],
        "concept_1": X_windows[:-1, 1, :],
        "concept_2": X_windows[:-1, 2, :]
    }

    Ys = X_windows[1:, :, -1]
    return Xs, Ys


if __name__ == "__main__":
    if (len(args) != ARGS_COUNT):
        usage()
    else:
        path_to_model = args[1]
        path_to_data_file = args[2]
        output_file = args[3]
        print(f"Loading model from {path_to_model}")
        model = keras.models.load_model(path_to_model)
        print(f"Loading data from {path_to_data_file}")
        Xs, Ys = load_data_from_path(path_to_data_file)
        print("Predicting...")
        predictions = model.predict(Xs)
        print(f"Saving predictions to {output_file}")
        np.savetxt(output_file, predictions, fmt="%f", delimiter=",")
        print("Done.")
