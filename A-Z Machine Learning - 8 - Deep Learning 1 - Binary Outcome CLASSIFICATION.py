#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Geodemographic segmentation Model (a Classification problem)
# show working directory
import os
#os.getcwd()
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 8 Deep Learning\Section 35 - Artificial Neural Networks (ANN)\Python')
os.getcwd()


# In[2]:


### Part-1 :
### Data Pre-Processing Template

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()


# In[5]:


x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print(x)


# In[8]:


# Encoding Categorical data - dummy ENCODING  - OneHotEncoding
# to Encode x :
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)
print(x[0:6, :])

# Avoiding dummy variable Trap
x = x[:, 1:]
print('x : \n',x[0:6, :])


# In[4]:


# Splitting the data into Test and Training set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[5]:


# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)


# In[7]:


### Part-2
### Create the DNN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the DNN
classifier = Sequential()

# Adding the Layers :
# 1 : input Layer + 1st hidden layer, 2 : 2nd hidden layer, 3: Output layer
#classifier.add(keras.layers.Flatten()) # input; if we needed Flatttening 
classifier.add(keras.layers.Dense(6, activation = 'relu', input_dim = 11))# inputLayer + 1st hidden layer
classifier.add(keras.layers.Dense(6, activation = 'relu'))    # 2nd hidden layer
classifier.add(keras.layers.Dense(1, activation = 'sigmoid')) # output layer; for Binary Classification
#                                                             # Note: if more than Binary;Ex: 3 classes
#                                                             # ==> units = 3, activation='softmax'

# Compiling our Model (= applying 'Stochastic Gradient Descent')
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#                                      # Note: if more than Binary;Ex: 3 classes
#                                      # ==> loss = categorical_crossentropy


# Fitting the ANN to the Training DATA-Set
classifier.fit(x = x_train, y = y_train, batch_size = 10, epochs = 50) # Note :
# batch_size = after what 'number of Observations' (used to train); do you want to UPDATE wheights ?
# epoch = after what 'number of Times' of passing DATA Set to the Model; do you want to STOP the training ?


# In[8]:


### Part 3 
### 3-1 : Predict on Test Set
y_pred = classifier.predict(x_test)


# In[20]:


### 3-2 : Evaluate with CONFUSION MATRIX; is the Model Validated on the Test Set ? 
# first: convert the Probability --> TRUE/FALSE
y_pred = (y_pred > 0.50)


# then : making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m) # finally leads to --> (1510 + 208 --> CORRECT Predictions)
#                                    (85 + 197   --> In-CORRECT Predictions)


# now : Compute Accuracy using Confusion Matrix 
accuracy_conf_m = (1510 + 208)/ 2000
print(accuracy_conf_m)  # accuracy_conf_m = 86 % ; which is Equal to Model Accuracy --> Valideted !


# In[68]:


dataset.head()


# In[ ]:




