import os
import sys
from inspect import getmembers, isfunction

import tensorflow as tf
from tensorflow import keras

import losses
from common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ARGS_COUNT = 3
WINDOW_SIZE = -1
exec_name, *args = sys.argv
CONCEPTS = -1


def usage():
    print("USAGE")
    print(f"{exec_name} [path_to_model] [path_to_data] [output_path]")
    print(f"Example: python3 {exec_name} /my/saved/model/dir /dataset/category1/data.csv /output/data_out.csv")


def window_gen(X, k):
    n = X.shape[0]
    for i in range(n - k):  # skipping last window
        window = X[i:i + k, :]
        yield [tuple(x) for x in window[:]]


def windows(X, k):
    return np.array(list(window_gen(X, k)))


def windows_to_inputs(X_windows):
    return {f"concept_{i}": X_windows[:, :, i] for i in range(CONCEPTS)}


def load_data(path, centers):
    df = load_file(path)
    data = to_6D_space(df)
    data = to_concepts(data, centers)
    windowed = windows(data, WINDOW_SIZE)
    return data[WINDOW_SIZE:], windows_to_inputs(windowed)


if __name__ == "__main__":
    if len(args) != ARGS_COUNT:
        usage()
        sys.exit(1)

    # parse arguments
    path_to_model, path_to_data_file, output_file = args

    # load model
    model = keras.models.load_model(path_to_model)

    # load fuzzy cluster centers
    centroids_path = os.path.join(path_to_model, 'centroids.csv')
    centroids = np.genfromtxt(centroids_path, delimiter=',')

    CONCEPTS = centroids.shape[0]
    WINDOW_SIZE = model.layers[0].input_shape[0][1]

    print(f"Loading data from '{path_to_data_file}'")
    in_concept_space, Xs = load_data(path_to_data_file, centroids)

    print("Predicting...")
    predictions = model.predict(Xs)

    border, header = 16, "Losses"
    print()
    print("#" * border, header, "#" * border)
    for func_name, func in getmembers(losses, isfunction):
        fmt_name = func_name.replace('_', ' ').capitalize()
        loss = func(np.float32(in_concept_space.flatten()), np.float32(predictions.flatten())).numpy()
        print("{:<30}\t{:<10.6f}".format(fmt_name, loss))

    print("#" * (border * 2 + len(header) + 2))
    print()

    output_name, output_ext = os.path.splitext(output_file)
    in_concept_space_name = f"{output_name}_concept_space{output_ext}"
    print(f"Saving input data in concept space to '{in_concept_space_name}'")
    np.savetxt(in_concept_space_name, in_concept_space, fmt="%f", delimiter=",")

    print(f"Saving predictions to '{output_file}'")
    np.savetxt(output_file, predictions, fmt="%f", delimiter=",")

    print("Done")
