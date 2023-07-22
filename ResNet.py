
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50


def load_train(path):
    params = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=.2,
        height_shift_range=.2
    )
    datagen = ImageDataGenerator(rescale=1/255, validation_split=.25, **params)
    label = pd.read_csv(path+ 'labels.csv')
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe = label,
        directory = path + 'final_files/',
        x_col = 'file_name',
        y_col = 'real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345) 

    return train_gen_flow

def load_test(path):
    params = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=.2,
        height_shift_range=.2
    )
    datagen = ImageDataGenerator(rescale=1/255, validation_split=.25, **params)
    label = pd.read_csv(path+ 'labels.csv')
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe = label,
        directory = path + 'final_files/',
        x_col = 'file_name',
        y_col = 'real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345) 

    return test_gen_flow


def create_model(input_shape):  # =(150, 150)):
    optimizer = Adam(learning_rate=.00001)

    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                    include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu')) 
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=15,
               steps_per_epoch=None, validation_steps=None):


    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model

