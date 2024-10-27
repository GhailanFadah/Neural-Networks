'''mnist.py
Loads and preprocesses the MNIST dataset
ghailan and Gordan
CS443: Bio-Inspired Machine Learning
Project 0: TensorFlow and MNIST
Spring 2024
'''
import os
import numpy as np
import tensorflow as tf


def get_mnist(N_val,max_value = 128,  path='data/mnist'):
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
    
    x_train_b = np.load(path+"/x_train.npy")
    y_train_b = np.load(path+"/y_train.npy")
    x_train, y_train, x_val, y_val = train_val_split(x_train_b, y_train_b, N_val)
    x_train_f = preprocess_mnist(x_train, max_value)
    x_test = np.load(path+"/x_test.npy")
    x_test_f = preprocess_mnist(x_test,max_value)
    y_test = np.load(path+"/y_test.npy")
    x_val_f = preprocess_mnist(x_val,max_value)
    
    
    # x_train_t = tf.constant(x_train_f, tf.float32)
    # y_train_t = tf.constant(y_train, tf.int64)
    # x_test_t = tf.constant(x_test_f, tf.float32)
    # y_test_t = tf.constant(y_test, tf.int64)
    # x_val_t = tf.constant(x_val_f, tf.float32)
    # y_val_t = tf.constant(y_val, tf.int64)
    
    return x_train_f, y_train, x_test_f, y_test, x_val_f, y_val


def preprocess_mnist(x, max_value = None):
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
    
    x_ = x/255
    
    x_f = np.reshape(x_, (x_.shape[0], x_.shape[1]*x_.shape[2]))
    if max_value: 
        x_f = x_f*max_value
    
  
    
    
    return x_f



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
    
    x_ = x[N_val:, :]
    y_ = y[N_val:]
    x_val = x[0:N_val, :]
    y_val = y[0:N_val]
    
    return x_, y_, x_val, y_val
