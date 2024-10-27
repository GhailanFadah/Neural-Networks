'''
Gordon Doore, Ghailan Fadah
Convolutional Encoder-Dense/KNN decoder network for classifying birdsong
04/18/2024
'''
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.models import Model

def conv_encode_decode(input_shape):
    '''
    '''
    #code based on that from project 1 extension
    #unit 1
    input_layer = tf.keras.Input(shape = input_shape)
    u1_conv = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(input_layer)
    u1_pool = layers.MaxPool2D(pool_size = (2,2))(u1_conv)
    #unit 2
    u2_conv = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(u1_pool)
    u2_pool = layers.MaxPool2D(pool_size = (2,2))(u2_conv)

    #unit 3
    u3_conv = layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu')(u2_pool)
    u3_pool = layers.MaxPool2D(pool_size = (2,2))(u3_conv)

    #decode with convolution:
    
    u4_conv = layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu')(u3_pool)
    u4_convT = layers.UpSampling2D((2,2))(u4_conv)
    #unit 4
    u5_conv = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(u4_convT)
    u5_convT = layers.UpSampling2D((2,2))(u5_conv)
    #unit 5
    u6_conv = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(u5_convT)
    u6_convT = layers.UpSampling2D((2,2))(u6_conv)
    
    u7_conv = layers.Conv2D(1, (3,3),padding = 'same', activation = 'relu')(u6_convT)

    model = models.Model(inputs = input_layer, outputs = u7_conv)

    return model


def conv_encode_classify(input_shape):
    '''  
    Last value should be the number of classes in the dataset
    regular convolutional classifier
    '''

    #code based on that from project 1 extension
    #unit 1
    input_layer = tf.keras.Input(shape = input_shape)
    u1_conv = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(input_layer)
    u1_pool = layers.MaxPool2D(pool_size = (2,2))(u1_conv)
    #unit 2
    u2_conv = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(u1_pool)
    u2_pool = layers.MaxPool2D(pool_size = (2,2))(u2_conv)

    #unit 3
    u3_conv = layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu')(u2_pool)
    u3_pool = layers.MaxPool2D(pool_size = (2,2))(u3_conv)

    #now predict classes using dense
    flat = layers.Flatten()(u3_pool)

    output_layer = layers.Dense(5,activation = 'softmax')(flat)
    model = models.Model(inputs = input_layer, outputs = output_layer)
    
    return model
