from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error 
import pandas as pd 
import numpy as np

# Data Preprocessing :
data = pd.read_csv('data.csv')
data = data.drop('pdiff',axis = 1)
data.dropna(inplace=True)
 
data['pdiff'] = data['pres'] - data['pprev']
data['month'] = data['month'].str[5:7].astype(int)


# Feature Extraction :
X = data.drop('pprev', axis=1).iloc[:-1]
Y = data['tavg'].iloc[1:]

# Scaling the features :
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train-test split           
split  = int(len(X)*0.8)               # weather data must be continuos and not randomly splitted(accuracy increased)
X_train = X[:split]
X_test = X[split:] 
Y_train = Y[:split]
Y_test = Y[split:]

# Model Training :
model = LinearRegression()                                                 # 86% accuracy
model.fit(X_train, Y_train)
weather = X_test

# model = RandomForestRegressor()                                              # 83% accuracy
# model.fit(X_train, Y_train)
# weather = X_test

# testing model accuracy :
predictions = model.predict(weather) 

predicted = []
actual = []
correct_prediction = 0

for temp in predictions :
    predicted.append(temp)
for temp in Y_test:
    actual.append(temp)

print(actual)
print([f"{x:.1f}" for x in predicted])

for i in range(len(predicted)):
    if (predicted[i]-actual[i]<=1) and (predicted[i]-actual[i]>=-1) :
        correct_prediction +=1
mae = mean_absolute_error(Y_test,predictions)
rmse = np.sqrt(mean_squared_error(Y_test,predictions))



print("Correct predictions = " , correct_prediction , " / " , len(predicted) , " with accuracy of " , (correct_prediction*100)/len(predicted))
print(mae)
print(rmse)