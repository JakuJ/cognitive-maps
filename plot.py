import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def getOptions(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("original", help="path to original data in concept space")
    parser.add_argument("predictions", help="path to predicted data")
    return parser.parse_args(args)


if __name__ == "__main__":
    # parse options
    options = getOptions(sys.argv[1:])

    # load data
    y_true = pd.read_csv(options.original, header=None).values
    y_pred = pd.read_csv(options.predictions, header=None).values

    # make a plot
    n, c = y_true.shape
    Xs = range(n)

    fig = plt.figure(figsize=(18, 6))
    axis = fig.subplots(1, c, sharey=True)
    axis[0].set_ylabel('y')

    for i, ax in enumerate(axis):
        ax.plot(Xs, y_true[:, i], 'b')
        ax.plot(Xs, y_pred[:, i], 'r')
        ax.set_title(f'Concept {i + 1}')
        ax.set_xlabel('x')

    plt.legend(['Original data', 'Predicted data'])
    plt.show()
