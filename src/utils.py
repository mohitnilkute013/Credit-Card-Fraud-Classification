import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.logger import logging
from src.exception import CustomException


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path, exist_ok=True)

        # opening the file and storing the object in it
        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error('Error in saving file object.')
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = {'Model_Name': [], "Model": [], "Acc_Score": [], "ConfusionMatrix": []}\
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            logging.info(f'Training on {model_name}')

            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_pred = model.predict(X_test)

            # r2 score
            test_score = accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)

            logging.info(f'Training Complete... Accuracy_Score: {test_score}')

            report['Model_Name'].append(model_name)
            report['Model'].append(model)
            report['Acc_Score'].append(test_score*100)
            report["ConfusionMatrix"].append(cm)

        return report
    
    except Exception as e:
        logging.error('Error in Training')
        raise CustomException(e, sys)



def load_object(filepath):
    try:
        with open(filepath) as file_obj:
            model = pickle.load(file = file_obj)

    except Exception as e:
        logging.error('Unable to read or load the file_obj')
        raise CustomException(e, sys)