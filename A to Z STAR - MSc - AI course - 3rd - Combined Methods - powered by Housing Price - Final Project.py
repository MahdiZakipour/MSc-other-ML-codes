#!/usr/bin/env python
# coding: utf-8

# In[3]:


# worked
# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 8 Deep Learning\Deep Learning for Droplet D by  Comsol')
os.getcwd()


# In[4]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import vpython as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Boston housing dataset
data = pd.read_csv('D:/Learning/Deep Learning A-Z course - my files/Section 8 Deep Learning/Deep Learning for Droplet D by  Comsol/NAMED_housing.csv')
prices = data['medv'] # This is our target to predict (Y)  
features = data.drop('medv', axis = 1) # Our initially, features


# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

data.head()


# In[4]:


data_new_temp = data.drop(['chas', 'rad', 'age', 'nox', 'indus', 'tax', 'Unnamed: 0'], axis = 1)
data_new_temp = data_new_temp.rename(columns = {'rm':'Q_c', 'lstat':'Q_d', 'ptratio':'frequency',
                                                'medv':'Diameter', 'crim':'visc_cont', 'zn' : 'regime' , 
                                               'dis' : 'visc_disp', 'b' : 'press'})
data_new_temp['Diameter'] = data_new_temp['Diameter']*10
data_new_temp['frequency'] = data_new_temp['frequency']*.02
data_new_temp.head()


# In[3]:


### Feature Observation
# making some Assumptions for our Data, We’ll find out if these assumptions are correct through the project.

### Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns # for making statistical graphics
#                       Plot pairwise relationships in a dataset; Histograms and Scatters TOGATHER :)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate and show pairplot
sns.pairplot(data, size=4)
#plt.tight_layout()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate and show pairplot
sns.pairplot(data, size=4, kind = 'kde',
             x_vars = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat' ],
             y_vars = ['medv'])


# In[6]:


# slice our data into new_data 
data_new = data.loc[:, ['rm', 'lstat', 'ptratio', 'medv']]
features_new = data_new.drop('medv', axis = 1) # Our initially, features

# Data Manipulation :
# rm , rstat -> Q_c , Q_d 
# 10*medv -> Diameter
# 0.02 * ptratio -> frequency
data_new['medv'] = data_new['medv']*10
data_new['ptratio'] = data_new['ptratio']*.02
data_new = data_new.rename(columns = {'rm':'Q_c', 'lstat':'Q_d', 'ptratio':'frequency', 'medv':'Diameter'})
data_new

get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate and show pairplot
sns.pairplot(data_new, size=4, kind = 'kde')


# In[11]:


# Export Data files

data_new_temp = data_new_temp[data_new_temp.index < 100]
comsol_data = data_new_temp.to_csv('D:/Learning/Deep Learning A-Z course - my files/Section 8 Deep Learning/Deep Learning for Droplet D by  Comsol/DATAfromComsol.csv')

data_new = data_new[data_new.index < 100]
ML_data = data_new.to_csv('D:/Learning/Deep Learning A-Z course - my files/Section 8 Deep Learning/Deep Learning for Droplet D by  Comsol/DATAforML.csv')

data_new


# In[7]:


# Calculate and show correlation matrix
cm = np.corrcoef(data_new.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=['rm', 'lstat', 'ptratio', 'medv'],
                xticklabels=['rm', 'lstat', 'ptratio', 'medv'])


# In[13]:


# Import 'train_test_split'
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features_new, prices, test_size=0.2)


# In[17]:


### The Neural Network Model
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

# build our model
model = Sequential()

model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


# In[18]:


# Train the model
model.fit(X_train.values, y_train.values, epochs=100, shuffle=True)


# In[22]:


### Predicting with The DEEP LEARNING Regression Model
y_pred = model.predict([[6.5, 20, 20]])
y_pred


# In[23]:


### Random forest to the Data set
# ITS BETTER FOR MORE THAN 1-DIMENSION PROBLESMs
# Create your regressor object
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, criterion = 'mse', random_state = 0) 
#                                 IMPORTANT : n_estimators: Number of trees
regressor.fit(X_train, y_train)

### Predicting with The Regression Model
y_pred = regressor.predict([[6.5, 20, 20]])
y_pred


# In[24]:


### SVR to the Data set
# Create your regressor object
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Gaussian, good for Non-Linear

regressor.fit(X_train , y_train)

### Predicting with The Regression Model

y_pred = regressor.predict([[6.5, 20, 20]])
y_pred


# In[ ]:





# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate and show pairplot
sns.pairplot(data_new_temp, size=3, kind = 'kde')


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns # for making statistical graphics
#                       Plot pairwise relationships in a dataset; Histograms and Scatters TOGATHER :)

# Calculate and show correlation matrix
cm = np.corrcoef(data_new.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=['Q_cont', 'Q_disp', 'Frequency', 'Diameter'],
                xticklabels=['Q_cont', 'Q_disp', 'Frequency', 'Diameter'])


# In[17]:


# Calculate and show correlation matrix
cm = np.corrcoef(data_new_temp.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                #cbar=True,
                #annot=True,
                square=True,
                #fmt='.2f',
                annot_kws={'size': 15})

### Note : data_new_temp --> total data
###        data_new      --> desirable data


# In[16]:


sns.pairplot(data_new_temp, kind = 'kde', size = 3,
             x_vars = ['Q_c', 'Q_d','frequency', 'visc_cont', 'regime' ,'visc_disp', 'press' ],
             y_vars = ['Diameter'])


# In[ ]:





# In[41]:


data_new

