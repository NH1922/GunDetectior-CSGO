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
                             input_shape=(128,128,3),
                             activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# second convolution layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),
                             input_shape=(100,100,3),
                             activation ='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 3,activation = 'softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=5
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r"E:\datasets\CSGO\train_set",
                                                 target_size = (128,128),
                                                 batch_size = 15,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r"E:\datasets\CSGO\test_set",
                                            target_size = (128, 128),
                                            batch_size = 15,
                                            class_mode = 'categorical')


classifier.summary()

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 1000,workers=3)


classifier.save(r"E:\datasets\models\GunDet_v4")

