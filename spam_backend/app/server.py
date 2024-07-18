#server.py contents
import pandas as pd
from fastapi import FastAPI
import numpy as np
import pickle

import os

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)
# Construct the absolute path to the pickle file
filename = os.path.join(script_dir, "spam_classifier_pipeline.pkl")

with open(filename, 'rb') as file:
    model = pickle.load(file)

class_names = np.array(['Ham','Spam'])

# Initialize the app as a FastAPI class instance
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Spam Model Classifier API'}

@app.post('/predict')
def predict(data: dict) -> str:
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the email to predict.
        e.g. {"email": "What is this email?"}

    Returns:
        dict: A dictionary containing the predicted class.
    """        
    
    prediction = model.predict(pd.Series(data['email']))
    class_name = class_names[prediction][0]
    return class_name