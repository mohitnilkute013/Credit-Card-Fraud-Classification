import os
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from ensemble_tabpfn import EnsembleTabPFN
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.logger import logging
from src.utils import save_object, load_object, evaluate_models
from src.exception import CustomException

from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(os.getcwd(), 'artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self):
        self._config = ModelTrainerConfig()

    def initiate_training(self, trainarr, testarr):
        try:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'SGDClassifier': SGDClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                # 'SVC linear': SVC(kernel='linear'),
                # 'SVC rbf': SVC(kernel='rbf'),
                # # 'NaiveBias':MultinomialNB(),
                # # 'KNNR':KNeighborsClassifier(n_neighbors=5),
            
                # 'RandomForest':RandomForestClassifier(random_state=42),
                # 'AdaBoost':AdaBoostClassifier(),
                # 'Gradient Boosting':GradientBoostingClassifier(),
                # 'XGB':xgb.XGBClassifier(),
                # 'BaggingSVC':BaggingClassifier(estimator=SVC()),
                'Ensemble TabPFN': TabPFNClassifier()
            }

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                trainarr[:,:-1],
                trainarr[:,-1],
                testarr[:,:-1],
                testarr[:,-1]
            )

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
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
                 filepath=self._config.trained_model_file_path,
                 obj=best_model
            )

            logging.info('Saved Best Model file')

        except Exception as e:
            logging.error('Error in Model Training.')
            raise CustomException(e, sys)


if __name__ == '__main__':

    di = DataIngestion()
    trainpath, test_path = di.initiate_ingestion()

    transformer = DataTransformation()
    trainarr, testarr = transformer.initiate_transformation(trainpath, test_path)

    trainer = ModelTrainer()
    trainer.initiate_training(trainarr, testarr)