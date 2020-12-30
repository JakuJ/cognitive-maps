import numpy as np
import tensorflow as tf
from sys import argv
from tensorflow import keras
# from keras import layers
# from tensorflow.keras import layers
# from tensorflow.keras.constraints import Constraint
# import tensorflow.keras.backend as K
import pandas as pd
import os
import skfuzzy as fuzz
from sklearn.preprocessing import minmax_scale
import losses

ARGS_COUNT = 4
window_size = -1
debug_args = ["scriptName", "drive/MyDrive/model", "dataset/UWaveGestureLibrary/Test/1/1001.csv", "OUT.csv"] 
args = debug_args if "--debug" in argv else argv
CONCEPTS = -1

def usage():
  print("USAGE")
  print(f"./{args[0]} [path_to_model] [path_to_data] [output_path]")
  print(f"Example: ./{args[0]} /my/saved/model/dir /dataset/category1/data.csv /output/data_out.csv")

def window_gen(X, k):
 n = X.shape[0]
 for i in range(n - k): # skipping last window
  window = X[i:i+k, :]
  yield [tuple(x) for x in window[:]]

def windows(X, k):
  return np.array(list(window_gen(X, k)))

def windows_to_inputs(X_windows):
    """Windows to Xs, Ys tuple"""
    return {f"concept_{i}": X_windows[:, :, i] for i in range(CONCEPTS)}

def load_data(path, centroids):
    def load_file(path):
        df = pd.read_csv(path, header=None)
        normalized = minmax_scale(df.values)
        df = pd.DataFrame(normalized)
        df.columns = ['x', 'y', 'z']
        return df

    def to_6D_space(X):
        dX = X[1:, :] - X[:-1, :]
        X = np.hstack([X[1:, :], dX])
        return minmax_scale(X)

    def to_concepts(X, centroids):
        X_fuzzy, *_ = fuzz.cluster.cmeans_predict(X.T, centroids, 2, error=0.005, maxiter=1000, init=None)
        return X_fuzzy.T

    df = load_file(path)
    data = to_6D_space(df.values)
    data = to_concepts(data, centroids)
    windowed = windows(data, window_size)
    return data[window_size:], windows_to_inputs(windowed)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if (len(args) != ARGS_COUNT):
        usage()
    else:
        path_to_model = args[1]
        path_to_data_file = args[2]
        output_file = args[3]
        print(f"Loading model from {path_to_model}")
        model = keras.models.load_model(path_to_model)
        centroids = np.genfromtxt(f"{path_to_model}/centroids.csv", delimiter=',')
        CONCEPTS = centroids.shape[0]
        window_size = model.layers[0].input_shape[0][1]
        print('window size', window_size)
        print(centroids)
        print(f"Loading data from {path_to_data_file}")
        #Xs, Ys = load_data_from_path(path_to_data_file)
        in_concepts, Xs = load_data(path_to_data_file,centroids)
        print("Predicting...")
        predictions = model.predict(Xs)
        print("predictions", predictions.shape)
        print("concepts",in_concepts.shape)
        # print(f"evaluate -> {model.evaluate(Xs)}")
        losses = losses.mean_squared_error(in_concepts.flatten(),predictions.flatten())
        print(f"losses -> {losses}")

        in_concepts_filename = f"{output_file}_in_concept_space"
        print(f"Saving  to {in_concepts_filename}")
        np.savetxt(in_concepts_filename, in_concepts, fmt="%f", delimiter=",")

        print(f"Saving predictions to {output_file}")
        np.savetxt(output_file, predictions, fmt="%f", delimiter=",")

        print("Done.")
