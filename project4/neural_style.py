'''neural_style.py
Use Neural Style Transfer to create an image with the style of the Style Image and the content of the Content Image
YOUR NAMES HERE
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2023
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import tf_util


class NeuralStyleTransfer:
    '''Runs the Neural Style Transfer algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, style_layer_names, content_layer_names):
        '''
        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        style_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the STYLE of the generated image.
        content_layer_names: Python list of str. len = 1.
            Names of layers in `pretrained_net` that we will readout netAct values. These netAct values contribute
            to the CONTENT of the generated image. We assume that this is a length 1 list (one selected layer str).

        TODO:
        1. Define instance variables for:
            - the pretrained network
            - the number of selected style layers used to readout netAct.
            - the number of selected content layers used to readout netAct.
        2. Make separate readout models for the selected style and content layers. Assign them as instance variables.
            This should be handled in a dedicated method in this class (`initialize_readout_models`).
        '''
        self.loss_history = None

        self.style_readout_model = None
        self.content_readout_model = None
        self.net = pretrained_net
        self.num_style_layers = len(style_layer_names)
        self.num_content_layers = len(content_layer_names)
        self.style_readout_model = self.initialize_readout_models(style_layer_names, content_layer_names)[1]
        self.content_readout_model = self.initialize_readout_models(style_layer_names, content_layer_names)[0]
        

    def initialize_readout_models(self, style_layer_names, content_layer_names):
        '''Creates and assigns style and content readout models to the instance variables self.style_readout_model and
        self.content_readout_model, respectively.

        Parameters:
        -----------
        style_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the STYLE of the generated image.
        content_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the CONTENT of the generated image.
        '''
        # Compute netAct values for selected layers with the style and content images
        readout_net_acts_con = []
        for layer in self.net.layers:
            if layer.name in content_layer_names:
                readout_net_acts_con.append(self.net.get_layer(layer.name).output)
                
        readout_net_acts_sty = []
        for layer in self.net.layers:
            if layer.name in style_layer_names:
                readout_net_acts_sty.append(self.net.get_layer(layer.name).output)
       
    
        return (tf.keras.Model(inputs=self.net.input, outputs=readout_net_acts_con), tf.keras.Model(inputs=self.net.input, outputs=readout_net_acts_sty))
        

    def gram_matrix(self, A):
        '''Computes the Gram matrix AA^T (<-- the ^T here means transpose of the A matrix on the right).

        Parameters:
        -----------
        A: tf tensor. shape=(K, blah). Matrix of which we want to compute the Gram matrix.

        Returns:
        -----------
        The Gram matrix of A. shape=(K, K).
        '''
        return A @ tf.transpose(A)
        

    def style_loss_layer(self, gen_img_layer_net_acts, style_img_layer_net_acts):
        '''Computes the contribution of the current layer toward the overall style loss.

        See notebook for equation for the loss contribution of a single layer.

        Parameters:
        -----------
        gen_img_layer_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the GENERATED IMAGE input at the CURRENT STYLE layer.
        style_img_layer_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the STYLE IMAGE input at the CURRENT STYLE layer.

        Returns:
        -----------
        The style loss contribution for the current layer. float.
        '''
      
        B, num_rows, num_cols, K = gen_img_layer_net_acts.shape.as_list()
        gen_img_layer_net_acts = tf.reshape(gen_img_layer_net_acts, shape=(3,25))
        style_img_layer_net_acts = tf.reshape(style_img_layer_net_acts, shape=(3,25))
        frac = 1/(2*(math.pow(K,2)*(math.pow(num_rows,2))* math.pow(num_cols,2)))
        print(frac)
        G = self.gram_matrix(style_img_layer_net_acts)-self.gram_matrix(gen_img_layer_net_acts)
        g_srt = tf.math.pow(G,2)
        sum_g = tf.reduce_sum(g_srt)
        print(sum_g)
        loss  =  frac*sum_g
        return loss

    def style_loss(self, gen_img_net_acts, style_img_net_acts):
        '''Computes the style loss — the average of style loss contributions across selected style layers.

        Parameters:
        -----------
        gen_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the GENERATED IMAGE input at the selected STYLE layers.
            Each item in the list g_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) generally differ in different layers of the network.
        style_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the STYLE IMAGE input at the selected STYLE layers.
            Each item in the list gen_img_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) generally differ in different layers of the network.

        Returns:
        -----------
        The overall style loss. float.
        '''
        pass

    def content_loss(self, gen_img_layer_act, content_img_layer_net_act):
        '''Computes the content loss.

        See notebook for the content loss equation.

        Parameters:
        -----------
        gen_img_layer_act: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the GENERATED IMAGE input at the CONTENT layer.
        content_img_layer_net_act: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the CONTENT IMAGE input at the CONTENT layer.

        Returns:
        -----------
        The content loss. float.
        '''
        B, num_rows, num_cols, K = gen_img_layer_act.shape.as_list()

    def total_loss(self, loss_style, style_wt, loss_content, content_wt):
        '''Computes the total loss, a weighted sum of the style and content losses.

        See notebook for the total loss equation.

        Parameters:
        -----------
        loss_style: float. Style loss.
        style_wt: float. Weighting factor for the style loss toward the total.
        loss_content: float. Content loss.
        content_wt: float. Weighting factor for the content loss toward the total.

        Returns:
        -----------
        The total loss. float.
        '''
        pass

    def forward(self, gen_img, style_img_net_acts, content_img_net_acts, style_wt, content_wt):
        '''Performs forward pass through pretrained network with the generated image `gen_img`. In addition, computes
        the image gradients and total loss based on the SELECTED content and style layers.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. Note that these are the raw PIXELS, NOT the netAct values.
        style_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the STYLE IMAGE input at the selected STYLE layers.
            Each item in the list gen_img_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) may differ in different layers of the network.
        content_img_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the CONTENT IMAGE input at the CONTENT layer.
        style_wt: float. Weighting factor for the style loss toward the total.
        content_wt: float. Weighting factor for the content loss toward the total.

        Returns:
        -----------
        loss. float. Sum of the total loss.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.

        TODO:
        While tracking gradients:
        - Use the readout model to extract the netAct values in the selected style layers and (separately) and content
        layer in response to `gen_img`.
        - Compute the style, content, and total loss.
        Then:
        - Obtain the tracked gradients of the loss with respect to the generated image.
        '''
        pass

    def fit(self, gen_img, style_img, content_img, n_epochs=200, style_wt=1e2, content_wt=1, lr=0.01,
            print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''Iteratively modify the generated image (`gen_img`) for `n_epochs` with the image gradients.
        In other words, run Neural Style Transfer on the generated image.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that will be modified across epochs.
            This is a trainable tf tensor (i.e. tf.Variable)
        style_img: tf tensor. shape=(1, Iy, Ix, n_chans). Style image. Used to derive the style that is applied to the
            generated image. This is a constant tf tensor.
        content_img: tf tensor. shape=(1, Iy, Ix, n_chans). Content image. Used to derive the content that is applied to
            generated image. This is a constant tf tensor.
        n_epochs: int. Number of epochs to run neural style transfer on the generated image.
        style_wt: float. Weighting factor for the style loss toward the total loss.
        content_wt: float. Weighting factor for the content loss toward the total loss.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `neural_style_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Get the netAct values of:
            - the style image at the selected style layers
            - the content image at the selected content layer
        1. Compute the forward pass on the generated image for `n_epochs`.
        2. Use the Adam optimizer to apply the gradients to the generated image.
            See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        3. Clip pixel values to the range [0, 1] and update the generated image.
            Reminder: Use `assign` rather than =.
        '''
        self.loss_history = []
