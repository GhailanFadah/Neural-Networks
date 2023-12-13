'''tf_util.py
Helper/utility functions related to using TensorFlow for Transfer Learning and working with images
YOUR NAMES HERE
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2023
'''
import numpy as np
from PIL import Image
import tensorflow as tf


def load_pretrained_net(net_name='vgg19'):
    '''Loads the pretrained network (included in Keras) identified by the string `net_name`.

    Parameters:
    -----------
    net_name: str. Name of pretrained network to load. By default, this is VGG19.

    Returns:
    -----------
    The pretained net. Keras object.

    NOTE: Pretrained net should NOT be trainable and NOT include the output layer.
    '''
    if net_name == 'vgg19':
        pretrained_net = tf.keras.applications.VGG19(include_top=False)
        pretrained_net.trainable = False
        return pretrained_net
    
    if net_name == 'xception':
        pretrained_net = tf.keras.applications.Xception(include_top=False)
        pretrained_net.trainable = False
        return pretrained_net
        
    
    
    


def get_all_layer_strs(pretrained_net):
    '''Gets the complete list of layer names from the pretrained net.

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network.

    Returns:
    -----------
    Python list of str. Length is the number of layers in the pretrained network.
    '''
    
    layer_names = []
    for layer in pretrained_net.layers:
        layer_names.append(layer.name)
    return layer_names
    


def filter_layer_strs(layer_names, match_str='conv4'):
    '''Extracts the layer name strs from `layer_names` (the complete list) that have `match_str` in the name.

    Parameters:
    -----------
    layer_names: Python list of str. The complete list of layer names in the pretrained network
    match_str: str. Substring searched for within each layer name

    Returns:
    -----------
    Python list of str. The list of layers from `layer_names` that include the string `match_str`
    '''
    desired_layers = []
    for layer in layer_names:
        if match_str in layer:
            desired_layers.append(layer)
            
    return desired_layers
    


def preprocess_image2tf(img, as_var):
    '''Converts an image (in numpy ndarray format) to TensorFlow tensor format

    Parameters:
    -----------
    img: ndarray. shape=(Iy, Ix, n_chans). A single image
    as_var: bool. Do we represent the tensor as a tf.Variable?

    Returns:
    -----------
    tf tensor. dtype: tf.float32. shape=(1, Iy, Ix, n_chans)

    NOTE: Notice the addition of the leading singleton batch dimension in the tf tensor returned.
    '''
    
    if as_var:
        raw = tf.Variable(img/255, dtype="float32", trainable=True)
        batch_img = tf.expand_dims(raw, 0)
        final_tf = tf.Variable(batch_img)
        
    else:
        raw = tf.constant(img/255, dtype="float32")
        batch_img = tf.expand_dims(raw, 0)
        final_tf =tf.constant(batch_img)
        
    return final_tf


def make_readout_model(pretrained_net, layer_names):
    '''Makes a tf.keras.Model object that returns the netAct (output) values of layers in the pretrained model
    `pretrained_net` that have the names in the list `layer_names` (the readout model).

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network
    layer_names: Python list of str. Selected list of pretrained net layer names whose netAct values should be returned
        by the readout model.

    Returns:
    -----------
    tf.keras.Model object (readout model) that provides a readout of the netAct values in the selected layer list
        (`layer_names`).
    '''
    
    readout_net_acts = []
    for layer in pretrained_net.layers:
        if layer.name in layer_names:
            readout_net_acts.append(pretrained_net.get_layer(layer.name).output)
       
    
    return tf.keras.Model(inputs=pretrained_net.input, outputs=readout_net_acts)


def tf2image(tensor):
    '''Converts a TensorFlow tensor into a PIL Image object.

    Parameters:
    -----------
    tensor: tf tensor. dtype=tf.float32. shape=(1, Iy, Ix, n_chans). A single image. Values range from 0-1.

    Returns:
    -----------
    PIL Image object. dtype=uint8. shape=(Iy, Ix, n_chans). Image representation of the input tensor with pixel values
        between 0 and 255 (unsigned ints).
        

    NOTE:
    - Scale pixel values to the range [0, 255] BEFORE converting to uint8 dtype.
    - Remove batch (singleton) dimension (if present)
    - One way to convert to PIL Image is to first convert to numpy ndarray.

    The following should be helpful:
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    '''
    # don't want to do tf.assign here because we would modify the gen_image in memory
    
    scale_ten = tensor*255
    #print(scale_ten)
    scale_ten = tf.squeeze(scale_ten)
    array = np.array(scale_ten).astype(np.uint8)
    
    img = Image.fromarray(array)
    
    return img
    pass
