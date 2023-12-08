import pickle
from flask import Flask, request, render_template, jsonify

import numpy as np
import pandas as pd


app= Flask(__name__)
#Loading model
logmodel = pickle.load(open('Heart_prediction.pkl' ,'rb'))
#loading scalar
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = logmodel.predict(new_data)
    print(output[0])
    # return jsonify(output[0])
    return jsonify(int(output[0]))

@app.route('/predict', methods = ['POST']) 
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = logmodel.predict(final_input)[0]
    
    if output == 0:
        output = "Absolutely Fine , No Possibilities of Heart Attack Detected"
    else :
        output = "Possibility of Heart Attack Detected ,Consider visiting a Doctor "
    
    
     
    return render_template("home.html",prediction_text="The condition of your heart: {}".format(output))
    

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0" ,port=5000)