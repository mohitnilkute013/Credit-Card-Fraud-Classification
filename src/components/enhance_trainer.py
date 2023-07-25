import os
import sys

import pandas as pd 
import numpy as np 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object, evaluate_models, enhance_model

from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


@dataclass
class EnhanceTrainerConfig:
    enhance_trainer_config_path = os.path.join(os.getcwd(), 'artifacts', 'enhanced_model.pkl')

class EnhanceTrainer:

    def __init__(self):
        self._config = EnhanceTrainerConfig()
    
    def initiate_training(self, trainarr, testarr):
        try:
            models = {
                'Gradient Boosting':GradientBoostingClassifier(),
            }

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                trainarr[:,:-1],
                trainarr[:,-1],
                testarr[:,:-1],
                testarr[:,-1]
            )

            param_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]
            }

            model_report = enhance_model(X_train, y_train, X_test, y_test, models, params = param_grid)

            print(pd.DataFrame(model_report))
            print('\n====================================================================================\n')
            logging.info(f'Model Report : \n{pd.DataFrame(model_report)}')

            # To get best model score from dictionary 
            index = list(model_report['Acc_Score']).index(max(model_report['Acc_Score']))

            best_model_score = model_report['Acc_Score'][index]
            best_model_name = model_report['Model_Name'][index]
            best_model = model_report['Model'][index]
            best_matrix = model_report['ConfusionMatrix'][index]


            print(f'Best Model Found , Model Name : {best_model_name} , Acc Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Acc Score : {best_model_score}')

            logging.info(f'Best Confusion Matrix: \n {ConfusionMatrixDisplay(best_matrix)}')

            save_object(
                 filepath=self._config.enhance_trainer_config_path,
                 obj=best_model
            )

            logging.info('Saved Best Model file')

        except Exception as e:
            logging.error('Error in Enhance Training.')
            raise CustomException(e, sys)


if __name__ == '__main__':

    di = DataIngestion()
    trainpath, test_path = di.initiate_ingestion()

    transformer = DataTransformation()
    trainarr, testarr = transformer.initiate_transformation(trainpath, test_path)

    trainer = EnhanceTrainer()
    trainer.initiate_training(trainarr, testarr)