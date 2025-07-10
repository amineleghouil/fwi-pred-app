from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        dc = float(request.form['DC'])
        isi = float(request.form['ISI'])
        bui = float(request.form['BUI'])
        region = request.form['Region']
        
    
        model = pickle.load(open('models/model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        
 
        features = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]])

        region_feature = 1 if region == 'bejaia' else 0
        features = np.append(features, [[region_feature]], axis=1)
  
        scaled_features = scaler.transform(features)
        
        prediction = model.predict(scaled_features)[0]
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__=="__main__":
    app.run(debug=True)
   
