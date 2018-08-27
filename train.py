# -*- coding: utf-8 -*-
"""
@author: NH
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
from matplotlib import pyplot as plt


classifier = Sequential()


classifier.add(Convolution2D(filters=32,kernel_size=(3,3),
                             input_shape=(64,64,3),
                             activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 32,activation='relu'))

#number of neurons = 2 for output layer because current dataset has 2 guns
classifier.add(Dense(units = 2,activation = 'softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=5
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r"E:\Guns\Training",
                                                 target_size = (100,100),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r"E:\Guns\Test",
                                            target_size = (100, 100),
                                            batch_size = 10,
                                            class_mode = 'categorical')


classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 1000,workers=3)
                         
classifier.save(r"GunDet")
