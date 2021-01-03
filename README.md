# Cognitive maps in time series prediction

This repository holds Python scripts responsible for training and evaluation of cognitive map–based ML models applied to
the problem of time series prediction.

## Requirements

* python 3.8
* pip

The dependencies can be installed with

```shell
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running

The `train.py` script is responsible for training a model on a set of training data (a folder of CSV files)
and saving it, along with the fuzzy cluster centers trained on this data, to a user-specified directory.

The `evaluate.py` script runs the serialized model on data in the provided CSV file and outputs files with the results,
as well as the inputs transformed into the concept space using centroids extracted from the training data.

Additionally, the `plot.py` script can be used to show a graphic comparison between the original data in concept space
and the predicted values.

To display detailed information about arguments for each script, run

```shell
    python3 <script> --help
```

An example run, using the "UWave Gesture Library" dataset might look like this:

```shell
  # train a model on the first category, with window width of 30, 3 features, 5 epochs, 3 concepts
  # and serialize the model to the "model" directory 
  python3 train.py -w 30 -f 3 -e 5 --train-source "./UWaveGestureLibrary/Train/1" --test-source "./UWaveGestureLibrary/Test/1" -c 3 --model-path model

  # evaluate the model on the 10.csv file in the 1st category, printing loss function values, 
  # saving results to output.csv, and writing concept-space input data to output_concept_space.csv
  python3 evaluate.py model UWaveGestureLibrary/Test/1/10.csv output.csv
  
  # show a plot comparing predicted and original data
  python3 plot.py output_concept_space.csv output.csv
```

Note that the files created by the `evaluate.py` script have `N - w - 1` rows, where `N` is the number of rows in the
input file and `w` is the window width. The `- 1` comes from skipping the last window of the provided data, as it
wouldn't correspond to any original data point (it would be a prediction of the hypothetical `N+1`th data point).