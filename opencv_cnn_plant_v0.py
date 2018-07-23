#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:37:24 2018

@author: mohit

"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm #to show progress
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, f1_score


#Load image 
image_dir_test = 'images/plants/test/'
image_dir_train = 'images/plants/train/'

#define the range for green color
sensitivity = 30
#define final image size
image_size = 64
'''
define a function to remove background from the image to only leave the green leaves. SUbsequenty transfer it to
gray scale, followed by resizing them to 64 x 64 size image

'''


def image_transformation(imageName, sensitivity):
    
    imagePlatCV = cv2.imread(imageName) #read image
    hsvImage = cv2.cvtColor(imagePlatCV, cv2.COLOR_BGR2HSV)
    #define the range for green color
    lower_green = np.array([60 - sensitivity, 100, 50])
    upper_green = np.array([60 + sensitivity, 255, 255])
    # threshold the hsv image to get only green colors
    mask = cv2.inRange(hsvImage, lower_green, upper_green)
    #apply bitwise_and between mask and the original image
    greenOnlyImage = cv2.bitwise_and(imagePlatCV, imagePlatCV, mask=mask)
    #lets define a kernal with ones
    kernel0 = np.ones((15,15), np.uint8)
    #lets apply closing morphological operation
    closing0 = cv2.morphologyEx(greenOnlyImage, cv2.MORPH_CLOSE, kernel0)
    #convert to gray scale
    grayScale = cv2.cvtColor(closing0, cv2.COLOR_BGR2GRAY)
    print(grayScale.shape)
    #blur the edges
    blurImage = cv2.GaussianBlur(grayScale, (15,15), 0)
    #resize image
    resizeImage = cv2.resize(blurImage, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resizeImage = resizeImage/255 #normalize
    resizeImage = resizeImage.reshape(64,64,1) #to make it in right dimensions for the Keras add 1 channel
    print(resizeImage.shape)
    return resizeImage




#define classes
classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed'
           , 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
'''
Data extraction: The loop below will create a data list containing image file path name, the classifcation lable (0 -11) and the specific plant name

'''

train = [] #data list
for species_lable, speciesName in enumerate(classes):
    for fileName in os.listdir(os.path.join(image_dir_train, speciesName)):
        train.append([image_dir_train + '{}/{}'.format(speciesName, fileName), species_lable, speciesName])
        
        
#convert the list into dataframe using Pandas

trainigDataFrame = pd.DataFrame(train, columns=['FilePath', 'PlantLabel', 'PlantName'])

#Suffle the data

seed = 1234
trainigDataFrame = trainigDataFrame.sample(frac=1, random_state=seed)
trainigDataFrame = trainigDataFrame.reset_index()

#Prepare the images for the model by preprocessing

X = np.zeros((trainigDataFrame.shape[0], image_size, image_size, 1)) #array with image size of each procssed image after image_transformfunction

for i, fileName in tqdm(enumerate(trainigDataFrame['FilePath'])):
    print(fileName)
    newImage = image_transformation(fileName, sensitivity)
    X[i] = newImage




#Convert lables to categorical and do one-hot encoding, followed by conversion to numpy array
y = trainigDataFrame['PlantLabel']
y = pd.Categorical(y)
y = pd.get_dummies(y)
y = y.values

#split dataset into Train and Test
X_train_dev, X_val, y_train_dev, y_val = train_test_split(X, y, test_size=0.10, random_state=seed)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size= 0.10, random_state=seed)



#generate more datasamples 

datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                            width_shift_range=0.2, # Range for random horizontal shifts
                            height_shift_range=0.2, # Range for random vertical shifts
                            zoom_range=0.2, # Range for random zoom
                            horizontal_flip=True, # Randomly flip inputs horizontally
                            vertical_flip=True) # Randomly flip inputs vertically
 
datagen.fit(X_train)


#define training model
def cnn_model():
    classifier = Sequential()
    classifier.add(Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), input_shape=(image_size, image_size,1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(image_size, image_size,1), activation='relu'))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(image_size, image_size,1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(image_size, image_size,1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Flatten())
    classifier.add(Dense(units=12, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier


batch_size = 64
epochs = 50
checkpoint = ModelCheckpoint('model3.h5', verbose=1, save_best_only=True)

model = cnn_model()

#trtain model
trainingModel = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=X_train.shape[0],
                           epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_dev, y_dev))

#final model
final_model = load_model("model3.h5")

final_loss, final_accuracy = final_model.evaluate(X_val, y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

#prediction
y_pred = final_model.predict(X_val)
y_pred = (y_pred>0.5)

#confusion matrix and classification report
y_pred = np.argmax(y_pred, axis=1)


cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))

print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1), target_names=classes))

#plot F1-score vs. Classes

f1Score = f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average=None)

y_pos = np.arange(len(classes))
plt.bar(y_pos, f1Score)
plt.xticks(y_pos, classes)
plt.ylabel('F1 Score')
plt.title('F1 Score of various species after classification')
plt.show()


