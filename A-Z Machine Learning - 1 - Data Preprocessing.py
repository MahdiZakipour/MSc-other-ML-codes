#!/usr/bin/env python
# coding: utf-8

# In[1]:


# working directory
import os
os.chdir('D:\Learning\Deep Learning A-Z course - my files\Section 1 Data Pre-processing\Section 3 - Data Preprocessing in Python\Python')
os.getcwd()

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# import data set
dataset = pd.read_csv('D:\Learning\Deep Learning A-Z course - my files\Section 1 Data Pre-processing\Section 3 - Data Preprocessing in Python\Python\Data.csv')
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
#print(y)


# In[3]:


# taking care of missing data /// Numercal
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

print(x)
print(y)


# In[4]:


#  Categorical data - dummy ENCODING  - OneHotEncoding
# to Encode x :
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])



from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

print(x)

# to Encode y :
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print(y)


# In[5]:


# splitting the data into Test and Training set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[6]:


# feature Scaling /// NORMALIZATION - or - STANDARADIZATION
from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train[:, 1:3] = scale_x.fit_transform(x_train[:, 1:3])
x_test[:, 1:3] = scale_x.transform(x_test[:, 1:3])


# In[7]:


### THIS BLOCK was ADDED, seperately. So be carefull with the names.

### Feature Observation
# making some Assumptions for our Data, We’ll find out if these assumptions are correct through the project.

### Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns # for making statistical graphics
#                       Plot pairwise relationships in a dataset; Histograms and Scatters TOGATHER :)

get_ipython().run_line_magic('matplotlib', 'inline')

# Calculate and show pairplot
sns.pairplot(data, size=2.5)
plt.tight_layout()


# In[8]:


x_train


# In[9]:


x_test


# In[23]:


onehotencoder

