#!/usr/bin/env python
# coding: utf-8

# In[1]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 2 Regression\Section 7 - Multiple Linear Regression\Python')
os.getcwd()


# In[2]:


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


# Encoding Categorical data - dummy ENCODING  - OneHotEncoding
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
"""""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""""

# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
""""from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train[:, 1:3] = scale_x.fit_transform(x_train[:, 1:3])
x_test[:, 1:3] = scale_x.transform(x_test[:, 1:3])"""


# In[3]:


### Fitting the Regression Model to the Data set
# Create your regressor object

### Predicting with The Regression Model
y_pred = regressor.predict()


### Visualising the Regression Results (if possible)
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Regression Model')
plt.xlabel('Position status')
plt.ylabel('Salary')
plt.show()


# In[1]:


### Visualising the Regression Results (for HIGHER RESOLUTION and smoother curve)
x_grid = np.arange(min(x), max(x), .1)
x_grid = x_grid.reshape(len(x_grid, 1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Regression Model')
plt.xlabel('Position status')
plt.ylabel('Salary')
plt.show()


# In[6]:





# In[ ]:




