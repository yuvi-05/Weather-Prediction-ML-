# Weather-Prediction-ML-
“Developed a machine learning-based weather prediction system to forecast next-day temperature and rainfall using historical weather data with Linear Regression and XGBoost.”

📌 Project Overview

This project focuses on predicting next day temperature and rainfall using historical and current weather data.
Machine Learning models such as Linear Regression and XGBoost are used to learn patterns from past weather conditions and make short-term predictions.

The goal of this project is to demonstrate how basic ML techniques can be applied to real-world weather data for forecasting.

🚀 Features

Predicts next day average temperature

Predicts next day rainfall

Uses historical + current day weather data

Implements multiple ML models for comparison

Simple and beginner-friendly implementation

🧠 Machine Learning Models Used

Linear Regression – Baseline model for temperature prediction

XGBoost Regressor – Advanced model for better accuracy and non-linear relationships

📊 Dataset & Data Source

Weather data is fetched using a free weather API that provides:

Average Temperature

Precipitation (Rainfall)

Atmospheric Pressure

Wind Speed

The model uses:

Previous day weather data

Current day weather data

to predict next day values.

⚙️ Tech Stack

Programming Language: Python

Libraries Used:

NumPy

Pandas

Matplotlib

Scikit-learn

XGBoost

Requests (for API calls)

🏗️ Project Workflow

Data Collection

Fetch weather data using API

Data Preprocessing

Handling missing values

Feature selection

Data formatting

Model Training

Train ML models on historical data

Prediction

Predict next day temperature & rainfall

Evaluation

Compare actual vs predicted values

📂 Project Structure
weather-prediction/
│

├── data/

│   └── weather_data.csv

│

├── train/

│   ├── train_test.py

│   └── model_training.py

│

├── models/

│   └── saved_models.pkl

│

├── requirements.txt

├── README.md

└── main.py


▶️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/weather-prediction.git


Install dependencies

pip install -r requirements.txt


Run the project


python main.py
