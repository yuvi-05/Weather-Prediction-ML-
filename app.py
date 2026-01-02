from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor

import pandas as pd 

import joblib
import os
import time
import requests

t1 = time.time()

# Importing Weather Data from Open-Meteo API
latitude = 18.5204    # Pune coordinates
longitude = 73.8567

url = "https://api.open-meteo.com/v1/forecast"                 # Open-Meteo API for historical and current weather data

params = {
    "latitude": latitude,
    "longitude": longitude,
    "daily": [
        "temperature_2m_mean",
        "precipitation_sum",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
    ],
    "hourly":[
        "cloud_cover"
    ],
    "past_days": 1,        # yesterday
    "forecast_days": 1,    # today
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()
# print (data)
# Calculating average cloud cover from hourly data
cloud = data["hourly"]["cloud_cover"]
avg_cloudcover = sum(cloud) / len(cloud)

# Create DataFrame for model input
df = pd.DataFrame({
    "date": data["daily"]["time"],
    "avg_temperature": data["daily"]["temperature_2m_mean"],
    "total_precipitation": data["daily"]["precipitation_sum"],
    "avg_pressure": data["daily"]["pressure_msl_mean"],
    "avg_wind_speed": data["daily"]["wind_speed_10m_mean"]  
})

print(df)
month = df["date"][1][5:7]             # current month
tprev = df["avg_temperature"][0]       # previous day's average temperature
tavg = df["avg_temperature"][1]        # current day's average temperature
prcp = df["total_precipitation"][1]    # current day's precipitation
wspd = df["avg_wind_speed"][1]         # current day's wind speed
pres = df["avg_pressure"][1]           # current day's pressure
pdiff = pres - df["avg_pressure"][0]   # difference in pressure from previous day
cc = avg_cloudcover                  # average cloud cover

# Data Preprocessing :
data = pd.read_csv('dataset/data.csv')  # meteostat dataset
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
    model = LinearRegression()                                                   # 87% accuracy(0.5 MAE and 0.7 RMSE)
    model.fit(X, Y)
    
    # saving the model :
    joblib.dump(model, 'model/weather_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')


# Input Data for Prediction :
weather = pd.DataFrame({
        'month': [month],              # current month
        'tprev': [tprev],              # previous day's average temperature
        'tavg': [tavg],                # current day's average temperature
        'prcp': [prcp],                # current day's precipitation
        'wspd': [wspd],                # current day's wind speed
        'pres': [pres],                # current day's pressure
        'cloudcvr': [cc],              # average cloud cover
        'pdiff': [pdiff]               # difference in pressure from previous day
        
    })

# Scaling the input features :
weather = scaler.transform(weather)

# Making Predictions :
predictions = model.predict(weather) 
print(f"Tomorrow's Temperature in Pune : {predictions.astype(float)[0]:.2f} °C")

t2 = time.time()
print(f"Execution Time: {t2 - t1} seconds")