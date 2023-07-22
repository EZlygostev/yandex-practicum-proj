#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path='/datasets/fruits_small/'):
    # параметры разделения и масштабирования данныхa
    ig_params = dict(
        validation_split=.25,
        rescale=1/255
    )
    # параметры аугментации
    augment_params = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=.2,
        height_shift_range=.2
    )
    # параметры получения изображений из источника
    flow_params = dict(
        directory=path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12354
    )
    # создание генератора обучающий изображений
    train_datagen = (
        ImageDataGenerator(**ig_params, **augment_params)
        .flow_from_directory(subset='training', **flow_params)
    )
    return train_datagen


def create_model(input_shape):  # =(150, 150)):
    optimizer = Adam(learning_rate=.001)
    model = Sequential()
    
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same', input_shape=input_shape, activation='relu'))
    model.add(AvgPool2D(pool_size=(3, 3), strides=None, padding='valid'))
        
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Conv2D(filters=46, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Flatten())
    # model.add(Dense(units=120, activation='relu'))
    # model.add(Dense(units=84, activation='relu'))
    
    model.add(Dense(units=12, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=5,
               steps_per_epoch=None, validation_steps=None):


    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model

