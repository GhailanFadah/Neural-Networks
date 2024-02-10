'''mnist.py
Loads and preprocesses the MNIST dataset
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 0: TensorFlow and MNIST
Spring 2024
'''
import os
import numpy as np
import tensorflow as tf


def get_mnist(N_val, path='data/mnist'):
    '''Load and preprocesses the MNIST dataset (train and test sets) located on disk within `path`.

    Parameters:
    -----------
    N_val: int. Number of data samples to reserve from the training set to form the validation set. As usual, each
    sample should be in EITHER training or validation sets, NOT BOTH.
    path: str. Path in working directory where MNIST dataset files are located.

    Returns:
    -----------
    x_train (training samples): tf.constant. tf.float32.
    y_train (training labels): tf.constant. tf.int64.
    x_test (test samples): tf.constant. tf.float32.
    y_test (test labels): tf.constant. tf.int64.
    x_val (validation samples): tf.constant. tf.float32.
    y_val (validation labels): tf.constant. tf.int64.

    NOTE: Using NumPy in this file is fine as long as you convert to the appropriate TensorFlow Tensor format by the end.
    '''
    pass


def preprocess_mnist(x):
    '''Preprocess the data `x` so that:
    - the maximum possible value in the dataset is 1 (and minimum possible is 0).
    - the shape is in the format: `(N, M)`

    Parameters:
    -----------
    x: ndarray. shape=(N, I_y, I_x). MNIST data samples represented as grayscale images.

    Returns:
    -----------
    ndarray. shape=(N, I_y*I_x). MNIST data samples represented as MLP-compatible feature vectors.
    '''
    pass


def train_val_split(x, y, N_val):
    '''Divide samples into train and validation sets. As usual, each sample should be in EITHER training or validation
    sets, NOT BOTH. Data samples are already shuffled.

    Parameters:
    -----------
    x: ndarray. shape=(N, M). MNIST data samples represented as vector vectors
    y: ndarray. ints. shape=(N,). MNIST class labels.
    N_val: int. Number of data samples to reserve from the training set to form the validation set.

    Returns:
    -----------
    x: ndarray. shape=(N-N_val, M). Training set.
    y: ndarray. shape=(N-N_val,). Training set class labels.
    x_val: ndarray. shape=(N_val, M). Validation set.
    y_val ndarray. shape=(N_val,). Validation set class labels.
    '''
    pass
