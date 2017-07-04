#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 03:22:19 2017

@author: adrianoyoshino
"""

# Convolutional Neural Network

#Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dropout
import os
import keras

# Initializing the CNN
classifier = Sequential()

# Creating Convolution layer
classifier.add(Conv2D(32, (5, 5), input_shape = (64, 64, 3), activation = 'relu'))

# Applying Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding dropout
classifier.add(Dropout(0.2))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('moedas/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('moedas/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'sparse')


classifier.fit_generator(training_set,
                         steps_per_epoch = 78,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 15)

# Evaluating model
classifier.evaluate_generator(test_set,
                              steps = 32)
classifier.metrics_names

vintecinco = test_datagen.flow_from_directory('moedas/test_set/te',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'sparse')

pred = classifier.predict_generator(vintecinco,
                              steps = 32)
c25 = []
for k in range(0, len(pred)):
    c25.append([i for i, j in enumerate(pred[k]) if j == max(pred[k])])
    
c25.count([2])



from keras.preprocessing import image
import numpy as np
test_image = image.load_img('moedas/moeda100.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_classes(test_image)
#training_set.class_indices
result

# serialize model to JSON
model_json = classifier.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model_final.h5")
print("Saved model to disk")