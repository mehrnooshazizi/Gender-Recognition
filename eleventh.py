# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:21:39 2020

@author: Poorvahab
"""
################################################ STEP 1

import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split

path_female='G:/Mehrrn0sh/DataSet/Gender Dataset/Daatset-merge/Train/Female/'
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

x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=None)

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

###################################################### STEP 2

from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout,BatchNormalization,MaxPool1D
from keras.regularizers import L2
from keras.losses import binary_crossentropy
from keras.optimizers import SGD

model=Sequential()
model.add(Conv1D(96,7,activation='relu',padding='same',input_shape=(100,100)))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=3))
model.add(Dropout(0.25))
model.add(Conv1D(256,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=3))
model.add(Dropout(0.25))
model.add(Conv1D(384,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=3))
model.add(Dropout(0.25))
model.add(Conv1D(384,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=3))
model.add(Dropout(0.25)) 
model.add(Conv1D(384,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=1))
model.add(Dropout(0.25)) 

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax',activity_regularizer=L2(0.0005)))

######################################################## STEP 3

model.compile(loss=binary_crossentropy,optimizer=SGD(lr=0.01, momentum=0.9),metrics=['accuracy'])
print('_____Training Machine_____')
print('_____Please Wait...!_____')

######################################################## STEP 4

net=model.fit(x_train,y_train,batch_size=258,epochs=50,verbose=1,validation_split=0.1)

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

loss,acc=model.evaluate(x_test,y_test)
print(f'loss is : {loss} accuracy is: {acc}')







