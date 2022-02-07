#!/usr/bin/env python
# coding: utf-8

# In[12]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 2 Regression\Section 6 - Simple Linear Regression\Python')
os.getcwd()


# In[20]:


### Data Pre-Processing Template

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the data into Test and Training set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
""""from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train[:, 1:3] = scale_x.fit_transform(x_train[:, 1:3])
x_test[:, 1:3] = scale_x.transform(x_test[:, 1:3])"""


# In[24]:


### Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  # creating an OBJECT of the class which we imported
regressor.fit(x_train, y_train) # fit/feed the data into the Machine


# In[26]:


### Predicting by the Machine
y_pred = regressor.predict(x_test)


# In[36]:


### Visualising the Results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs experience (on Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs experience (on Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# In[28]:


y_test


# In[ ]:




