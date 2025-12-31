from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd 

import joblib
import os
import time

t1 = time.time()

# Data Preprocessing :
data = pd.read_csv('dataset/data.csv')
data = data.drop('pdiff',axis = 1)
data.dropna(inplace=True)
 
data['pdiff'] = data['pres'] - data['pprev']
data['month'] = data['month'].str[5:7].astype(int)

# Feature Extraction :
X = data.drop('pprev', axis=1).iloc[:-1]
Y = data['tavg'].iloc[1:]

if os.path.exists('model/weather_model.pkl') and os.path.exists('model/scaler.pkl'):
    # Loading the model and scaler :
    model = joblib.load('model/weather_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

else:
    # Scaling the features :
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model Training :
    model = LinearRegression()
    model.fit(X, Y)
    
    # saving the model :
    joblib.dump(model, 'model/weather_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')


# Input Data for Prediction :
weather = pd.DataFrame({
        'month': [3],              # current month
        'tprev': [27.5],           # previous day's average temperature
        'tavg': [28.2],            # current day's average temperature
        'prcp': [0.0],             # current day's precipitation
        'wspd': [6.8],             # current day's wind speed
        'pres': [1008.5],          # current day's pressure
        'pdiff': [-0.9]            # difference in pressure from previous day
    })

# Scaling the input features :
weather = scaler.transform(weather)

# Making Predictions :
predictions = model.predict(weather) #29.9
print(predictions.astype(float))

t2 = time.time()
print(f"Execution Time: {t2 - t1} seconds")