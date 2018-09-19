import sys
sys.path.append('../../')
import pandas as pd

from src.ds import analysis
from src import helpers


def main():
    train_x ,train_y = helpers.load_data()

    analysis.model_comparison(train_x,train_y)


if __name__ == '__main__':
    main()
