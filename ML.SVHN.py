#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the required Keras modules containing model and layers
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import scipy.io as sio
from scipy.io import loadmat
from sklearn.model_selection import train_test_split



img_rows, img_cols = 32, 32
#image_index = img

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

X_train, y_train = load_data('SVHN_data/train_32x32.mat')
X_test, y_test = load_data('SVHN_data/test_32x32.mat')

print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

print("Training Set", X_train.shape)
print("Test Set", X_test.shape)
print('')
input_shape = (32, 32, 3)

# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)


# In[ ]:


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape)) #Convolved with the 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu)) #y = x1*w1 + b
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x = X_train,y= y_train, epochs=15)

model.evaluate(X_test, y_test)


# In[ ]:


#Test prediction

img = 9999
plt.imshow(x_test[img].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[img].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())

