import sys
sys.path.append('../../')
import pandas as pd

from src.ds import analysis


def main():
    train = pd.read_csv('../../resources/data/titanic_train.csv')
    print(train.columns)
    train = train.drop(['SibSp','Parch','Sex','Fare','Name','Embarked',
                                                'Initial','Age'],axis=1)
    x_cols = ['Fare_Bin','Age_Bin','Initial_Bin','Sex_Bin','Embarked_Bin',
            'Family_size','is_alone']
    y_cols = 'Survived_Bin'
    train_x = train[x_cols]
    train_y = train[y_cols]

    analysis.model_comparison(train_x,train_y)


if __name__ == '__main__':
    main()
