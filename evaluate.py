import os
import sys
from inspect import getmembers, isfunction

from tensorflow import keras

import losses
from common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def usage(name):
    print("USAGE")
    print(f"[interpreter] {name} [path_to_model] [path_to_data] [output_path]")
    print(f"Example: python3 {name} /my/saved/model/dir /dataset/category1/data.csv /output/data_out.csv")


def load_data(path, centers, num_concepts, window_width):
    df = load_file(path)
    data = expand_derivatives(df)
    data = to_concepts(data, centers)
    windowed = windows(data, window_width, skip_last=True)
    return data[window_width:], windows_to_inputs(windowed, num_concepts, skip_last=False)


if __name__ == "__main__":
    # parse arguments
    exec_name, *args = sys.argv
    if len(args) != 3:
        usage(exec_name)
        sys.exit(1)

    path_to_model, path_to_data_file, output_file = args

    # load model
    model = keras.models.load_model(path_to_model)

    # load fuzzy cluster centers
    centroids_path = os.path.join(path_to_model, 'centroids.csv')
    centroids = np.genfromtxt(centroids_path, delimiter=',')

    concepts = centroids.shape[0]
    window_size = model.layers[0].input_shape[0][1]

    print(f"Loading data from '{path_to_data_file}'")
    in_concept_space, Xs = load_data(path_to_data_file, centroids, concepts, window_size)

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
