import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load the preprocessed data
go = pd.read_csv('Goldprice1.csv')

# Select only numeric columns for regression
numeric_go = go.select_dtypes(include=[float, int])

# Splitting the features and target
X = numeric_go.drop(columns=['GLD'])
y = numeric_go['GLD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Random Forest Regressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Predict on the test set
pred = reg.predict(X_test)

# Calculate R^2 score
score = r2_score(y_test, pred)

# Streamlit web app elements
st.title('Gold Price Model')
img = Image.open('img.jpeg')
st.image(img)
st.subheader('Using RandomForest')
st.write(go)
st.subheader('Model Performance')
st.write(f'R^2 Score: {score}')
