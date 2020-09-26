import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import urllib
import joblib


app = Flask(__name__)
model = joblib.load('homepred.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    
    
    prediction = model.predict(final_features)
    
    output = ""
    if prediction == 0:
        output = " Customer is not Interested"
    else:
        output = "Customer is Interested"



    return render_template('home.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
