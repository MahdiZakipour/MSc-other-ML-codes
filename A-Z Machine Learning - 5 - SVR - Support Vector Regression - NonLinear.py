#!/usr/bin/env python
# coding: utf-8

# In[1]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 2 Regression\Section 9 - Support Vector Regression (SVR)\Python')
os.getcwd()


# In[6]:


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

#feature Scaling /// NORMALIZATION - or - STANDARADIZATION
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_y = StandardScaler()
x = scale_x.fit_transform(x)
y = scale_y.fit_transform(y.reshape(-1, 1))


# In[3]:


### Fitting SVR to the Data set
# Create your regressor object
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Gaussian, good for Non-Linear

regressor.fit(x , y)


# In[4]:


### Predicting with The Regression Model
y_pred = scale_y.inverse_transform(regressor.predict(scale_x.transform(np.array([[6.5]]))))


# In[5]:


### Visualising the Regression Results (if possible)
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position status')
plt.ylabel('Salary')
plt.show()


# In[9]:


### Visualising the Regression Results (for HIGHER RESOLUTION and smoother curve)
x_grid = np.arange(min(x), max(x), .1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Regression Model')
plt.xlabel('Position status')
plt.ylabel('Salary')
plt.show()


# In[ ]:




