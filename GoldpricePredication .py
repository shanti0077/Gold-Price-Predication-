import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[2]:


go = pd.read_csv('Goldprice1.csv')


# In[3]:


go.head    


# In[4]:


#SPX= is a free float weighted measurement stock market index of 500 largest companies listed on stock exchange 
#GLD= gold price 
#USO = united states oil fund 
#SLV= sliver 
#EUR/USD= money value 


# In[6]:


go.shape   #6 feature 2290 colums in data set


# In[7]:


go.info()


# In[8]:


#checking missing values 
go.isnull().sum()   # no null


# In[10]:


go.describe() # check statitics  #mean count std min etc


# In[18]:


# Assuming 'go' is your DataFrame
# Convert date columns to datetime if needed
go['date_column'] = pd.to_datetime(go['Date'], errors='coerce')

# Select only numeric columns
numeric_go = go.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlate = numeric_go.corr()

print(correlate)


# In[20]:


plt.figure(figsize= (6,6))
sns.heatmap(correlate,annot= True)


# In[23]:


print(correlate['GLD'])


# In[30]:


sns.displot(go['GLD'], color= "orange")


# In[66]:


#Splitting the feature and Target 
X= correlate.drop(['Date', 'GLD'], axis= 1)
y= correlate['GLD']


# In[65]:


x.shape


# In[36]:


y.shape #gld dependented on date other are independent 


# In[74]:


#train_test_split (type)+shift +tab
#X: The features (input data). y: The target values (output data).test_size=0.33: This specifies the proportion of the dataset to include in the test split. Here, 33% of the data will be used for testing, and 67% for training.random_state=42: This is a seed value to ensure that the split is reproducible. By setting a random_state, you ensure that you get the same split every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[68]:


print(X_train.dtypes)


# In[70]:


X_train = X_train.select_dtypes(include=[float,int])
X_test = X_test.select_dtypes(include=[float,int])


# In[72]:


reg = RandomForestRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)
print(f'R^2 Score: {score}')


# In[73]:


print(pred)


# In[ ]:




