﻿from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K

from data2 import load_train_data, load_test_data
from keras.utils.visualize_util import plot

img_rows = 176
img_cols = 176

def get_unet():
    inputs = Input((2, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    model.compile(optimizer=Adam(lr=1e-5), loss='mse')

    return model

def train_and_predict():
    imgs_train, imgs_output_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_output_train = imgs_output_train.astype('float32')
    imgs_output_train /= 255.  # scale outputs to [0, 1]

    imgs_test, imgs_output_test = load_test_data()

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    
    imgs_output_test = imgs_output_test.astype('float32')
    imgs_output_test /= 255.  # scale outputs to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', save_best_only=True )

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    earlyStopping=EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    model.fit(imgs_train, imgs_output_train, batch_size=16, nb_epoch=100, verbose=1, callbacks=[model_checkpoint, earlyStopping], validation_split=0.1, validation_data=None, shuffle=True)
        
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting outputs on test data...')
    print('-'*30)
    imgs_output_test2 = model.predict(imgs_test, verbose=1)
    np.save('imgs_output_test2.npy', imgs_output_test2)

    print('-'*30)
    print('Evaluating test loss...')
    print('-'*30)
    score = model.evaluate(imgs_test, imgs_output_test, batch_size=16)
    print(score)

    print('-'*30)
    print('Ploting model...')
    print('-'*30)
    plot(model, to_file='model.png')


if __name__ == '__main__':
    train_and_predict()