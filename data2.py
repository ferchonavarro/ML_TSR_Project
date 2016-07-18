from __future__ import print_function

import os
import numpy as np
import re
import cv2

data_path = 'raw2/'

image_rows = 176
image_cols = 176

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = sorted(os.listdir(train_data_path), key=natural_keys)
    total = len(images)

    imgs = np.ndarray((total - 2 * (total / 208), 2, image_rows, image_cols), dtype=np.uint8)
    imgs_output = np.ndarray((total - 2 * (total / 208), 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    j = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = np.array([img])
        if i % 208 < 206:
            imgs[j,0] = img
        if i % 208 > 0 and i % 208 < 207:
            imgs[j-1,1] = img
        if i % 208 > 1:
            imgs_output[j-2] = img

        if (i+1) % 100 == 0:
            print('Done: {0}/{1} images'.format(i+1, total))
        i += 1
        j += 1
        if i % 208 == 0:
            j -= 2
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_output_train.npy', imgs_output)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_output_train = np.load('imgs_output_train.npy')
    return imgs_train, imgs_output_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = sorted(os.listdir(train_data_path), key=natural_keys)
    total = len(images)

    imgs = np.ndarray((total - 2 * (total / 208), 2, image_rows, image_cols), dtype=np.uint8)
    imgs_output = np.ndarray((total - 2 * (total / 208), 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.dtype("S16"))

    i = 0
    j = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if i % 208 > 1:
            img_id = str(image_name.split('.')[0])
            imgs_id[j-2] = img_id
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = np.array([img])
        if i % 208 < 206:
            imgs[j,0] = img
        if i % 208 > 0 and i % 208 < 207:
            imgs[j-1,1] = img
        if i % 208 > 1:
            imgs_output[j-2] = img

        if (i+1) % 100 == 0:
            print('Done: {0}/{1} images'.format(i+1, total))
        i += 1
        j += 1
        if i % 208 == 0:
            j -= 2
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_output_test.npy', imgs_output)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_output_test = np.load('imgs_output_test.npy')
    return imgs_test, imgs_output_test

def load_imgs_id():
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
