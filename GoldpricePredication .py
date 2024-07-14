import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# Load the data
go = pd.read_csv('Goldprice1.csv')

# Display the first few rows of the dataframe
st.write(go.head())

# Display the shape of the dataframe
st.write(f'DataFrame Shape: {go.shape}')  # 6 features, 2290 rows

# Display basic information about the dataframe
st.write(go.info())

# Check for missing values
st.write(go.isnull().sum())  # no null values

# Display statistical summary of the dataframe
st.write(go.describe())  # mean, count, std, min, etc.

# Convert date columns to datetime if needed
go['Date'] = pd.to_datetime(go['Date'], errors='coerce')

# Select only numeric columns for correlation calculation
numeric_go = go.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlate = numeric_go.corr()
st.write(correlate)

# Plot the correlation matrix
plt.figure(figsize=(6, 6))
sns.heatmap(correlate, annot=True)
st.pyplot(plt)

# Display the correlation values for 'GLD'
st.write(correlate['GLD'])

# Plot distribution of 'GLD'
sns.displot(go['GLD'], color="orange")
st.pyplot(plt)

# Splitting the feature and target
X = numeric_go.drop(columns=['GLD'])
y = numeric_go['GLD']

# Ensure the target variable is correctly displayed
st.write(f'Target variable shape: {y.shape}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Display data types of training features
st.write(X_train.dtypes)

# Train a Random Forest Regressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Predict on the test set
pred = reg.predict(X_test)

# Calculate R^2 score
score = r2_score(y_test, pred)

# Streamlit web app
st.title('Gold Price Model')
img = Image.open('img.jpg')
st.image(img)
st.subheader('Using RandomForest')
st.write(go)
st.subheader('Model Performance')
st.write(f'R^2 Score: {score}')
streamlit run GoldpricePredication.py
