import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from sys import argv
from tensorflow import keras
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import minmax_scale
import losses
from inspect import getmembers, isfunction


ARGS_COUNT = 4
WINDOW_SIZE = -1
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

    df = load_file(path)
    data = to_6D_space(df.values)
    data = to_concepts(data, centroids)
    windowed = windows(data, WINDOW_SIZE)
    return data[WINDOW_SIZE:], windows_to_inputs(windowed)


if __name__ == "__main__":
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
        WINDOW_SIZE = model.layers[0].input_shape[0][1]
        print(f"Loading data from {path_to_data_file}")
        in_concept_space, Xs = load_data(path_to_data_file,centroids)
        print("Predicting...")
        predictions = model.predict(Xs)
        print("#"*15, "Losses", "#"*15)
        [tf.print(func_name,func(np.float32(in_concept_space.flatten()),np.float32(predictions.flatten()))) for func_name, func in getmembers(losses, isfunction)]
        print("#" * (15*2 + len(" Losses ")))
        print()
        in_concept_space_filename = f"{output_file}_concept_space"
        print(f"Saving input data in concept space to {in_concept_space_filename}")
        np.savetxt(in_concept_space_filename, in_concept_space, fmt="%f", delimiter=",")

        print(f"Saving predictions to {output_file}")
        np.savetxt(output_file, predictions, fmt="%f", delimiter=",")

        print("Done.")
