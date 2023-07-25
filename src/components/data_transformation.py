import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self):

        num_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

        num_pipeline = Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, num_cols),
            ('cat_pipeline', cat_pipeline, cat_cols)
        ])

        return preprocessor


    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            # Target column
            target_column = 'default payment next month'
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            logging.info('Obtaining Preprocessing object...')

            preprocessor = self.get_preprocessor()
            # Preprocessing data
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info('Preprocessing completed with input training and testing dataset...')

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            logging.info('Saving Preprocessor file...')

            save_object(
                filepath=self.transformation_config.preprocessor_path,
                obj=preprocessor
            )

            return train_arr, test_arr

        except Exception as e:
            logging.error('Error in Data Transformation.')
            raise CustomException(e, sys)


if __name__ == '__main__':

    di = DataIngestion()
    trainpath, test_path = di.initiate_ingestion()

    transformer = DataTransformation()
    print(transformer.initiate_transformation(trainpath, test_path))