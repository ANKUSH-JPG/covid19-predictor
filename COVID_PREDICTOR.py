#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential


# In[2]:


from keras.layers import MaxPool2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Convolution2D


# In[5]:


from keras.layers import Dense


# In[6]:


from keras.optimizers import Adam


# In[ ]:





# In[7]:


model=Sequential()


# In[8]:


model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))


# In[9]:


model.add(MaxPool2D(pool_size=(2,2)))


# In[10]:


model.add(Flatten())


# In[11]:


model.summary()


# In[12]:


model.add(Dense(units=1024,activation='relu'))


# In[13]:


model.add(Dense(units=512,activation='relu'))


# In[14]:


model.add(Dense(units=250,activation='relu'))


# In[15]:


model.add(Dense(units=75,activation='relu'))


# In[16]:


model.add(Dense(units=25,activation='relu'))


# In[17]:


model.add(Dense(units=10,activation='relu'))


# In[18]:


model.add(Dense(units=1,activation='sigmoid'))


# In[ ]:





# In[19]:


model.summary()


# In[20]:


model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:





# In[21]:


from keras_preprocessing.image import ImageDataGenerator


# In[22]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_dataset = train_datagen.flow_from_directory(
        'F:/MLOPS/practise/ENVIRONMENT2/xray_dataset_covid19/train/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
test_dataset = test_datagen.flow_from_directory(
        'F:/MLOPS/practise/ENVIRONMENT2/xray_dataset_covid19/test/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
model.fit(
        train_dataset,
        steps_per_epoch=500,
        epochs=4,
        validation_data=test_dataset,
        validation_steps=800)


# In[ ]:





# In[24]:


model.save('covid19_predictor.h5')


# In[25]:


train_dataset.class_indices


# In[1]:


from keras.models import load_model
from keras.preprocessing import image
import numpy


# In[2]:


loaded_model=load_model('covid19_predictor.h5')


# In[8]:


a=image.load_img('F:/MLOPS/practise/ENVIRONMENT2/xray_dataset_covid19/test/NORMAL PERSON/NORMAL2-IM-0132-0001.jpeg',
                               target_size=(64,64))


# In[9]:


image_test=image.img_to_array(a)


# In[10]:


image_test.shape
image_test=numpy.expand_dims(image_test,axis=0)


# In[11]:


test=loaded_model.predict(image_test)


# In[12]:


if test[0][0]==0.0:
    print("INFECTED PERSON")

else:
    print('NORMAL PERSON')


# In[ ]:




