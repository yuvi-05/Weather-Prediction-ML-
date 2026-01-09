# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error , mean_squared_error 
import pandas as pd 
import numpy as np

# Data Preprocessing :
data = pd.read_csv('dataset/data.csv')
data = data.drop('pdiff',axis = 1)
data.dropna(inplace=True)
 
data['pdiff'] = data['pres'] - data['pprev']
# month in CSV is like '01-01-2023 00:00' -> slice positions 3:5 give the month
data['month'] = data['month'].str[3:5].astype(int)


# Feature Extraction :
X = data.drop('pprev', axis=1).iloc[:-1]
Y = data[['tempmin','tavg','tempmax','prcp']].iloc[1:]

# Scaling the features :
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train-test split           
split  = int(len(X)*0.8)               # weather data must be continuos and not randomly splitted(accuracy increased)
X_train = X[:split]
X_test = X[split:] 
Y_train = Y[:split]
Y_test = Y[split:]

# # Model Training :
# model = LinearRegression()                                                   # 86% accuracy(0.7 MAE and 1 RMSE)
# model.fit(X_train, Y_train)
# weather = X_test

# model = RandomForestRegressor()                                              # 82% accuracy(0.5 MAE and 0.7 RMSE)
# model.fit(X_train, Y_train)
# weather = X_test

model = XGBRegressor(                                                          # 85% accuracy (0.5 MAE and 0.7 RMSE)
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.8
)
model.fit(X_train, Y_train)
weather = X_test

# testing model accuracy :
predictions = model.predict(weather) 
predictions_df = pd.DataFrame(predictions,columns =['tempmin','tavg','tempmax','prcp'])

predicted = []
actual = []
correct_prediction = 0

for temp in predictions_df['rcp'] :
    predicted.append(float(temp))
    
for temp in Y_test['prcp'] :
    actual.append(temp)


# print(predicted)

for i in range(len(predicted)):
        
    if predicted[i] < 0.1 :                   # for precipitation values less than 0.1 consider it as 0
        predicted[i] = 0.0    
    if actual[i] > 8 :                        # heavy rainfall values capped at 8
        actual[i] = 8.0
    if predicted[i] > 8 :
        predicted[i] = 8.0        
    if (predicted[i]-actual[i]<=1) and (predicted[i]-actual[i]>=-1) :
        correct_prediction +=1

print(actual)
print([f"{x:.1f}" for x in predicted])

mae = mean_absolute_error(Y_test,predictions)
rmse = np.sqrt(mean_squared_error(Y_test,predictions))



print("Correct predictions = " , correct_prediction , " / " , len(predicted) , " with accuracy of " , (correct_prediction*100)/len(predicted))
print(mae)
print(rmse)