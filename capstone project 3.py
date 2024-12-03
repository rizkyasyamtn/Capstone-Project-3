
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading dataset...")
try:
    data = pd.read_csv('C:/Users/USER/Documents/Purwadhika Boothcamp/Capstone Project 3/dataset/california_housing.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please ensure 'california_housing.csv' is in the same directory.")
    exit()

print("\nDataset Info:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

print("\nHandling missing values...")
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())
print("Missing values handled.")

print("\nEncoding categorical data...")
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
print("Categorical encoding completed.")

print("\nSplitting features and target...")
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling completed.")

print("\nSplitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data splitting completed.")

print("\nTraining the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

print("\nMaking predictions and evaluating the model...")
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

print("\nSaving the trained model...")
joblib.dump(model, 'california_housing_model.pkl')
print("Model saved as 'california_housing_model.pkl'.")
