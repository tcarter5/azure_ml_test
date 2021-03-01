# -*- coding: utf-8 -*-

import json
import pickle

from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

features = [
    'age', 
    'sex', 
    'chest_pain', 
    'blood_pressure', 
    'serum_cholestoral',
    'fasting_blood_sugar', 
    'electrocardiographic', 
    'max_heart_rate',
    'induced_angina', 
    'ST_depression', 
    'slope', 
    'vessels', 
    'thal',
]

# Load saved ML models
tf_model = load_model('neural')
with open('adaboost.pkl', 'rb') as f:
    adaboost_model = pickle.load(f)
    

# Load scaler info    
with open('scaler_means.json') as fin:
    scaler_means = json.load(fin)
    
with open('scaler_sigmas.json') as fin:
    scaler_sigmas = json.load(fin)


# Define function to scale json data passed to endpoint
def scale_data(data_json):
    
    for key in data_json:
        data_json[key] = (data_json[key] - scaler_means[key])/scaler_means[key]
    
    return data_json


# Convert scaled json data to a numpy array
def convert_to_array(data_dict):
    
    myarray = [data_dict[key] for key in features]
    myarray = np.array(myarray)
    
    return myarray.reshape(-1, len(features))


# make app
app = Flask(__name__)
app.config["DEBUG"] = True

# define endpoints
@app.route('/', methods=['GET'])
def home():       
            
    return 'App is Healthy'


@app.route('/adaboost', methods=['POST'])
def adaboost():       
        
    content = scale_data(request.json)
    data_array = convert_to_array(content)
    prediction = int(adaboost_model.predict(data_array))
    
    return jsonify(prediction)


@app.route('/neural', methods=['POST'])
def neural():       
        
    content = scale_data(request.json)
    data_array = convert_to_array(content)
    prediction = int((tf_model.predict(data_array) > 0.5).astype(int))
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run()