#!/usr/bin/env python
# coding: utf-8

# In[2]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 2 Regression\Section 7 - Multiple Linear Regression\Python')
os.getcwd()


# In[3]:


### Data Pre-Processing Template

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('50_Startups.csv')
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#  Categorical data - dummy ENCODING  - OneHotEncoding
# to Encode x :
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
print('x \n',x)

# Avoiding dummy variable Trap
x = x[:, 1:]
print('x \n',x)

# Splitting the data into Test and Training set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
""""from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train[:, 1:3] = scale_x.fit_transform(x_train[:, 1:3])
x_test[:, 1:3] = scale_x.transform(x_test[:, 1:3])"""


# In[4]:


### Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[5]:


### Predicting by the Machine
y_pred = regressor.predict(x_test)


# In[18]:


### Building Optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones(shape = (50,1)).astype(int), values = x, axis = 1) # add a column of ONEs to the beginning of the x (-> x_0), b_0*x_0
x_optimal = x[:, [0,1,2,3,4,5]]

regressor_OLS = sm.ols(y, x_optimal, data = )


# In[15]:


x


# In[ ]:




