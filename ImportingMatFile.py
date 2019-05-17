#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image_ind = 9999
train_data = sio.loadmat('Desktop/train_32x32.mat')

# access to the dict
x_train = train_data['X']
y_train = train_data['y']

# show sample
plt.imshow(x_train[:,:,:,image_ind])
plt.show()


print(y_train[image_ind])
print()


# In[ ]:





# In[ ]:




