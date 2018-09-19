import sys
sys.path.append('../../')
import pandas as pd

from src.ds import visualize as viz
from src import helpers


def main():

    train_x,train_y = helpers.load_data()
    viz.corr_heatmap(train_x)


if __name__ == '__main__':
    main()
