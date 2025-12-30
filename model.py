from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np

data = pd.read_csv('data.csv')
data = data.drop('pdiff',axis = 1)
data.dropna(inplace=True)
 
data['pdiff'] = data['pres'] - data['pprev']
data['month'] = data['month'].str[5:7].astype(int)

# print(data)

X = data.drop('pprev', axis=1).iloc[:-1]
# print (X)

Y = data['tavg'].iloc[1:]
# print(Y)

model = LinearRegression()
model.fit(X, Y)
weather = pd.DataFrame({
    'month': [3],              # current month
    'tprev': [27.5],           # previous day's average temperature
    'tavg': [28.2],            # current day's average temperature
    'prcp': [0.0],             # current day's precipitation
    'wspd': [6.8],             # current day's wind speed
    'pres': [1008.5],          # current day's pressure
    'pdiff': [-0.9]            # difference in pressure from previous day
})
predictions = model.predict(weather) #29.9
print(predictions.astype(float))