# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:54:24 2020

@author: Markazi.co
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
 
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.3,random_state=None)

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

###################################################### STEP 2

from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

model=Sequential()
model.add(Conv1D(64,3,activation='relu',padding='same',strides=2,input_shape=(100,100)))
model.add(Dropout(0.2))
model.add(Conv1D(128,3,activation='relu',strides=2,padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(256,3,activation='relu',strides=2,padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(512,3,activation='relu',strides=2,padding='same'))
model.add(Dropout(0.2)) 
model.add(Conv1D(1024,3,activation='relu',strides=2,padding='same')) 

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2,activation='sigmoid'))

######################################################## STEP 3

model.compile(loss=binary_crossentropy,optimizer=Adam(),metrics=['accuracy'])
print('_____Training Machine_____')
print('_____Please Wait...!_____')

######################################################## STEP 4

net=model.fit(x_train,y_train,batch_size=258,epochs=70,verbose=1,validation_split=0.2)


model.save_weights('model_wights.h5')
model.save('model.h5')

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







