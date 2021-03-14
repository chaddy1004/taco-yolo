from functools import partial

import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def _random_rotate(img_arr, max_angle):
    image_arr = ndimage.rotate(img_arr, np.random.uniform(-max_angle, max_angle), reshape=False)
    return np.array(image_arr)


def _gaussain_noise(img_arr, var=0.1):
    sigma = var ** 0.5
    mean = 0
    gaussian = np.random.normal(mean, sigma, img_arr.shape)
    img_noisy = img_arr + gaussian
    return img_noisy


def create_dataset(config):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = normalize_img(x_train)
    x_test = normalize_img(x_test)
    y_train_oh = scalar_to_onehot(y_train, 10)
    y_test_oh = scalar_to_onehot(y_test, 10)
    image_train = tf.data.Dataset.from_tensor_slices(x_train[..., np.newaxis])
    label_train = tf.data.Dataset.from_tensor_slices(y_train_oh)
    image_test = tf.data.Dataset.from_tensor_slices(x_test[..., np.newaxis])
    label_test = tf.data.Dataset.from_tensor_slices(y_test_oh)

    def tf_random_rotate(img_tensor):
        im_shape = img_tensor.shape
        random_rotate = partial(_random_rotate, max_angle=config.data.test.rotation.max_angle)
        [image_tensor, ] = tf.py_function(random_rotate, [img_tensor], [tf.float32])
        image_tensor.set_shape(im_shape)
        return image_tensor

    if config.data.augmentation:
        gaussian_noise_aug = partial(_gaussain_noise, var=0.5)
        image_train = image_train.map(tf_random_rotate)
        image_train = image_train.map(gaussian_noise_aug)

    if config.data.test.rotation.apply:
        image_test = image_test.map(tf_random_rotate)

    elif config.data.test.noise.apply:
        gaussian_noise = partial(_gaussain_noise, var=config.data.test.noise.variance)
        image_test = image_test.map(gaussian_noise)

    train_dataset = tf.data.Dataset.zip((image_train, label_train))
    test_dataset = tf.data.Dataset.zip((image_test, label_test))

    train_dataset = train_dataset.batch(config.trainer.batch_size)
    test_dataset = test_dataset.batch(config.trainer.batch_size)

    return train_dataset, test_dataset
