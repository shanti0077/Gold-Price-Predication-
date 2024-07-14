import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from PIL import Image 


go = pd.read_csv('Goldprice1.csv')

go.head    

#SPX= is a free float weighted measurement stock market index of 500 largest companies listed on stock exchange 
#GLD= gold price 
#USO = united states oil fund 
#SLV= sliver 
#EUR/USD= money value 

go.shape   #6 feature 2290 colums in data set

go.info()

#checking missing values 
go.isnull().sum()   # no null

go.describe() # check statitics  #mean count std min etc

# Assuming 'go' is your DataFrame
# Convert date columns to datetime if needed
go['date_column'] = pd.to_datetime(go['Date'], errors='coerce')

# Select only numeric columns
numeric_go = go.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlate = numeric_go.corr()

print(correlate)

plt.figure(figsize= (6,6))
sns.heatmap(correlate,annot= True)


print(correlate['GLD'])

sns.displot(go['GLD'], color= "orange")

#Splitting the feature and Target 
X= correlate.drop(['Date', 'GLD'], axis= 1)
y= correlate['GLD']

x.shape


y.shape #gld dependented on date other are independent 


#train_test_split (type)+shift +tab
#X: The features (input data). y: The target values (output data).test_size=0.33: This specifies the proportion of the dataset to include in the test split. Here, 33% of the data will be used for testing, and 67% for training.random_state=42: This is a seed value to ensure that the split is reproducible. By setting a random_state, you ensure that you get the same split every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print(X_train.dtypes)

X_train = X_train.select_dtypes(include=[float,int])
X_test = X_test.select_dtypes(include=[float,int])


reg = RandomForestRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)

#webapp
st.title('Gold Price Model')
img= Image.open('img.jpeg')
st.image(img)
st.subheader('Using RandomForest')
st.write(go)
st.subheader('Model Performance')
st.write(score)






