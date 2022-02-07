#!/usr/bin/env python
# coding: utf-8

# In[10]:


# worked
# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 8 Deep Learning\Deep Learning for Droplet D by  Comsol')
os.getcwd()


# In[11]:


### Part-1 :
### Data Pre-Processing Template

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# import data set
train_df  = pd.read_csv('D:/Learning/Deep Learning A-Z course - my files/Section 8 Deep Learning/Deep Learning for Droplet D by  Comsol/NAMED_housing.csv')
#data_frame = data_frame.values

train_df.head()


# In[19]:


target = 'medv'

### Normalizing / Scaling the Data ...
# MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] .
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[13], scaler.min_[13]))
multiplied_by = scaler.scale_[13]
added = scaler.min_[13]

scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)
scaled_train_df  = scaled_train_df.loc[:, ['rm', 'lstat', 'ptratio', 'medv']] # Normalized Data
scaled_train_df   # We choosed tha Data columns like here, according to the "file 3rd.ipynb" Explorations


# In[ ]:





# In[21]:


import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

# build our model
model = Sequential()

model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])


# In[22]:


X = scaled_train_df.drop(target, axis=1).values
Y = scaled_train_df[[target]].values

# Train the model
model.fit(
    X[10:],
    Y[10:],
    epochs=100,
    shuffle=True
)


# In[27]:


# predict
prediction = model.predict(X[:1])
y_0 = prediction[0][0]
print('Prediction with scaling - {}',format(y_0))
y_0 -= added
y_0 /= multiplied_by
print("Housing Price Prediction  - ${}".format(y_0))


# In[31]:


print(X[:1])
print(prediction)


# In[ ]:




