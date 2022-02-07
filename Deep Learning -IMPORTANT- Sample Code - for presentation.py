#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
import numpy as np


# In[17]:


### Data - Model
mnist = tf.keras.datasets.mnist # 28*28 images of numbers
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # split daraset to train and test

x_train = tf.keras.utils.normalize(x_train, axis=1)  # Normalizing/ scaling x_train
x_test = tf.keras.utils.normalize(x_test, axis=1)  # Normalizing/ scaling x_test


### Model Creation
model = tf.keras.models.Sequential() # sequential type, is the most common type of modek
model.add(tf.keras.layers.Flatten()) # 1st layer = INPUT layer /// 
                                     # also Flatten : from []28*28 ---> []1*784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 2nd layer // 128 Neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 3rd layer // 128 Neurons

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # 4th/OUTPUT layer // 10 units
                                        # activation : because it is a PROBABILITY, we use softmax activation function


# In[19]:


##### Train the Model on TRAINING Set
### compile ---> dtermine the whole process of Forward and BAckward propagation
# optmzr = tf.keras.optimizer.Adam(learning_rate='0.02')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # optimizer : adam is the most common, others : SGD, RMSprop, ...
                                    # loss : categorical_crossentropy is the most common (or binary_crossentropy)
                                    # metric : sth to be Evaluated by the Model


### Fit ---> fit the DNN Model to Data set
model.fit(x_train, y_train, epochs=8)       # Feeding the Model to TRAIN it.
                                            # batch_size : number of Observations after which, the Model Updates the wheights
                                            # epoch : a round of training on the Whole Data set
# (+++) check for Overfitting
#val_loss, val_acc = model.evaluate(x_test, y_test)  # To check for NOT Memorizing (overfitting)
                                                    # and to Generalize (find the patterns)
#print(val_loss, val_acc)                            #because it's not IDENTICAL nor TOO FAR, not Overfitted. :)
# (+++)



### make Predictions on the TEST Set (new Observations)
#predictions = model.predict(x_test)        #  Predicting
#print(predictions)
#print(predictions[343])

#import numpy as np
#print(np.argmax(predictions[343]))

#plt.imshow(x_test[343])
#plt.show()

### just in case :
#model.save('deeplearning_ExpertSys.model')
#new_model = tf.keras.models.load_model('deeplearning_ExpertSys.model')
#predictions = new_model.predict(x_test)





# In[23]:


### make Predictions on the TEST Set (new Observations)
predictions = model.predict(x_test)        #  Predicting
#print(predictions)
print(predictions[125])
plt.imshow(x_test[125])
plt.show()



#print(x_train[0])
#print(predictions)
#np.shape(predictions)
#np.shape(x_train)
#print(predictions[341])
#plt.imshow(x_test[341])
#plt.show()


# In[14]:


import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.show()


# In[26]:


print (28*28)
print (1875+313)


# In[6]:


print (x_train[2917])


# In[9]:


tf.keras.optimizers


# In[ ]:




