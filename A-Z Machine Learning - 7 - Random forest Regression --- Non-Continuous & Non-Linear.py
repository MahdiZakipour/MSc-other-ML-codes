#!/usr/bin/env python
# coding: utf-8

# In[1]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 2 Regression\Section 11 - Random Forest Regression\Python')
os.getcwd()


# In[2]:


# Random Forest Regression is Non-Linear & Non-continuos;
# BETTER FOR MORE THAN 1-DIMENSION PROBLESMs
# *** GOOD MODEL in Regression generally; because it's kind of ENSEMBLE LEARNING (like D.N.N) !

### Data Pre-Processing Template

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
""""from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train[:, 1:3] = scale_x.fit_transform(x_train[:, 1:3])
x_test[:, 1:3] = scale_x.transform(x_test[:, 1:3])"""


# In[3]:


### Fitting Random forest to the Data set
# ITS BETTER FOR MORE THAN 1-DIMENSION PROBLESMs
# Create your regressor object
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, criterion = 'mse', random_state = 0) 
#                                 IMPORTANT : n_estimators: Number of trees
regressor.fit(x, y)


# In[4]:


### Predicting with The Regression Model
y_pred = regressor.predict([[6.5]])
y_pred


# In[5]:


### Visualising the Regression Results (for HIGHER RESOLUTION and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Regression Model (Random forest Regression) \n now OK !')
plt.xlabel('Position status')
plt.ylabel('Salary')
plt.show()


# In[6]:


x


# In[7]:


y


# In[ ]:




