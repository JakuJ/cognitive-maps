import argparse
import os
import sys
from inspect import getmembers, isfunction

from tensorflow import keras

import losses
from common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getOptions(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model", help="path to the folder with model data")
    parser.add_argument("source", help="path to data file")
    parser.add_argument("output", help="name for the output file")
    return parser.parse_args(args)


def load_data(path, centers, num_concepts, window_width):
    df = load_file(path)
    data = expand_derivatives(df)
    data = to_concepts(data, centers)
    windowed = windows(data, window_width, skip_last=True)
    return data[window_width:], windows_to_inputs(windowed, num_concepts, skip_last=False)


if __name__ == "__main__":
    # parse arguments
    options = getOptions(sys.argv[1:])

    # load model
    model = keras.models.load_model(options.model, compile=False)
    model.compile(loss='mse', optimizer='sgd')

    # load fuzzy cluster centers
    centroids_path = os.path.join(options.model, 'centroids.csv')
    centroids = np.genfromtxt(centroids_path, delimiter=',')

    concepts = centroids.shape[0]
    window_size = model.layers[0].input_shape[0][1]

    print(f"Loading data from '{options.source}'")
    in_concept_space, Xs = load_data(options.source, centroids, concepts, window_size)

    print("Predicting...")
    predictions = model.predict(Xs)

    border, side_length, header = '-', 22, "Losses"
    print()
    print(border * side_length, header, border * side_length)

    for func_name, func in getmembers(losses, isfunction):
        fmt_name = func_name.replace('_', ' ').capitalize()
        loss = func(np.float32(in_concept_space.flatten()), np.float32(predictions.flatten())).numpy()
        print("{:<40}  {:>10.6f}".format(fmt_name, loss))

    print(border * (side_length * 2 + len(header) + 2))
    print()

    output_name, output_ext = os.path.splitext(options.output)
    in_concept_space_name = f"{output_name}_concept_space{output_ext}"
    print(f"Saving input data in concept space to '{in_concept_space_name}'")
    np.savetxt(in_concept_space_name, in_concept_space, fmt="%f", delimiter=",")

    print(f"Saving predictions to '{options.output}'")
    np.savetxt(options.output, predictions, fmt="%f", delimiter=",")

    print("Done")
