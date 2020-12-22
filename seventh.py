# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:53:40 2020

@author: Poorvahab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:54:24 2020

@author: Markazi.co
"""
################################################ STEP 1

import numpy as np
import glob
import cv2

path_female='G:/Mehrrn0sh/DataSet/Gender Dataset/Dataset-Origin/Train/Female/'
path_male='G:/Mehrrn0sh/DataSet/Gender Dataset/Daatset-merge/Train/Male/'
Male=glob.glob(path_male+'*.jpg')
Female=glob.glob(path_female+'*.jpg')

images_male=[]
labels_male=[]
images_female=[]
labels_female=[]
for x in Male:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float16')
    img=img/np.max(img)
    images_male.append(img)
    labels_male.append(0)
for x in Female:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float16')
    img=img/np.max(img)
    images_female.append(img)
    labels_female.append(1)    
    
images_male.extend(images_female)
labels_male.extend(labels_female)

images=np.array(images_male)
labels=np.array(labels_male)

from keras.utils import np_utils
labels=np_utils.to_categorical(labels)

path_female_test='G:/Mehrrn0sh/DataSet/Gender Dataset/Dataset-Origin/Test/Female/'
path_male_test='G:/Mehrrn0sh/DataSet/Gender Dataset/Dataset-Origin/Test/Male/'
Male_test=glob.glob(path_male_test+'*.jpg')
Female_test=glob.glob(path_female_test+'*.jpg')

images_male_test=[]
labels_male_test=[]
images_female_test=[]
labels_female_test=[]
for x in Male_test:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float16')
    img=img/np.max(img)
    images_male_test.append(img)
    labels_male_test.append(0)
for x in Female_test:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float16')
    img=img/np.max(img)
    images_female_test.append(img)
    labels_female_test.append(1)    
    
images_male_test.extend(images_female_test)
labels_male_test.extend(labels_female_test)

images_test=np.array(images_male_test)
labels_test=np.array(labels_male_test)

from keras.utils import np_utils
labels_test=np_utils.to_categorical(labels_test)

###################################################### STEP 2

from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout
from keras.regularizers import L2
from keras.losses import Hinge
from keras.optimizers import SGD

model=Sequential()
model.add(Conv1D(64,5,activation='relu',padding='same',strides=4,input_shape=(100,100)))
#model.add(Dropout(0.2))
model.add(Conv1D(128,3,activation='relu',strides=2,padding='same'))
#model.add(Dropout(0.2))
model.add(Conv1D(256,3,activation='relu',strides=2,padding='same'))
#model.add(Dropout(0.2))
model.add(Conv1D(512,3,activation='relu',strides=1,padding='same'))
#model.add(Dropout(0.2)) 
model.add(Conv1D(1024,3,activation='relu',strides=1,padding='same')) 

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2,activation='linear',activity_regularizer=L2(0.005)))

######################################################## STEP 3

model.compile(loss=Hinge(),optimizer=SGD(),metrics=['accuracy'])
print('_____Training Machine_____')
print('_____Please Wait...!_____')

######################################################## STEP 4

net=model.fit(images,labels,batch_size=258,epochs=20,verbose=1,validation_split=0.3)


#model.save_weights('model_wights.h5')
#model.save('model.h5')

def PlotModel(net):
    import matplotlib.pyplot as plt 
    history=net.history
    Accuracy=history['accuracy']
    ValidationAccuracy=history['val_accuracy']
    Loss=history['loss']
    ValidatioLoss=history['val_loss']

    plt.figure('Accuracy Diagram')
    plt.plot(Accuracy)
    plt.plot(ValidationAccuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Data','Validation Data'])
    plt.title('Accuracy Diagram')
    plt.show()

    plt.figure('Loss Diagram')
    plt.plot(Loss)
    plt.plot(ValidatioLoss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Data','Validation Data'])
    plt.title('Loss Diagram')
    plt.show()

PlotModel(net)

loss,acc=model.evaluate(images_test,labels_test)
print(f'loss is : {loss} accuracy is: {acc}')







