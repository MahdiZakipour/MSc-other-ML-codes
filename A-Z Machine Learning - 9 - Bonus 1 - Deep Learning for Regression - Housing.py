#!/usr/bin/env python
# coding: utf-8

# In[1]:


# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 8 Deep Learning\Deep Learning for Regression - Housing')
os.getcwd()


# In[2]:


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[3]:


# Load the Dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

dataframe.head()


# In[4]:


# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]


# In[11]:


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
    return model


# In[13]:


1


# In[14]:


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[13]:


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[5]:


### from This section, from another example :
# https://github.com/Prajwal10031999/House-Price-Prediction-with-ANN/blob/main/house-price-prediction-with-ann.ipynb

from keras.models import Sequential
from keras.layers import BatchNormalization  #####
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint #we can control our model if going well during validation part or not
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
import tensorflow as tf



model = Sequential()
model.add(Dense(units=256, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=128, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=8, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='linear'))


# In[6]:


model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics = ['accuracy'])
model.fit(X, Y, epochs=25, batch_size=64)
model.predict(X_test, batch_size=64)


# In[9]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




