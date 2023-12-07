'''deep_dream.py
Generate art with a pretrained neural network using the DeepDream algorithm
YOUR NAMES HERE
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2023
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt

import tf_util


class DeepDream:
    '''Runs the DeepDream algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, selected_layers_names):
        '''
        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        selected_layers_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values
            from in order to contribute to the generated image.

        TODO:
        1. Define instance variables for the pretrained network and the number of selected layers used to readout netAct.
        2. Make an readout model for the selected layers (use function in `tf_util`) and assign it as an instance variable.
        '''
        self.loss_history = None
        self.net = pretrained_net
        self.numLayers = len(selected_layers_names)
        self.readoutModel = tf_util.make_readout_model(pretrained_net, selected_layers_names)
        

    def loss_layer(self, layer_net_acts):
        '''Computes the contribution to the total loss from the current layer with netAct values `layer_net_acts`. The
        loss contribution is the mean of all the netAct values in the current layer.

        Parameters:
        -----------
        layer_net_acts: tf tensor. shape=(1, Iy, Ix, K). The netAct values in the current selected layer. K is the
            number of kernels in the layer.

        Returns:
        -----------
        loss component from current layer. float. Mean of all the netAct values in the current layer.
        '''
        return tf.reduce_mean(layer_net_acts)
        pass

    def forward(self, gen_img, standardize_grads=True, eps=1e-8):
        '''Performs forward pass through the pretrained network with the generated image `gen_img`.
        Loss is computed based on the SELECTED layers (in readout model).

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. The singleton dimension is the batch dimension (N).
        standardize_grads: bool. Should we standardize the image gradients?
        eps: float. Small number used in standardization to prevent possible division by 0 (i.e. if stdev is 0).

        Returns:
        -----------
        loss. float. Sum of the loss components from all the selected layers.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.

        TODO:
        While tracking gradients:
        - Use the readout model to extract the netAct values in the selected layers for `gen_img`.
        - Compute the average loss across all selected layers.
        Then:
        - Obtain the tracked gradients of the loss with respect to the generated image.
        '''
        
        
        with tf.GradientTape(persistent=True) as tape:
            netActs = self.readoutModel(gen_img)
        
            loss = 0
            for act in netActs:
                loss += self.loss_layer(act)
                
            loss = loss/ self.numLayers
            
                
        grads = tape.gradient(loss, gen_img)
        
        
        if standardize_grads:
            grads = (grads - tf.math.reduce_mean(grads))/(tf.math.reduce_std(grads) + eps)
            
        return loss, grads
            
        
        
       

    def fit(self, gen_img, n_epochs=26, lr=0.01, print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''Iteratively modify the generated image (`gen_img`) for `n_epochs` with the image gradients using the
            gradient ASCENT algorithm. In other words, run DeepDream on the generated image.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one. For example, image_1.jpg, image_2.jpg,
            etc.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Compute the forward pass on the generated image for `n_epochs`.
        2. Apply the gradients to the generated image using the gradient ASCENT update rule.
        3. Clip pixel values to the range [0, 1] and update the generated image.
            The TensorFlow `assign` function is helpful here because = would "wipe out" the tf.Variable property,
            which is not what we want because we want to track gradients on the generated image across epochs.
        4. After the first epoch completes, always print out how long it took to finish the first epoch and an estimate
        of how long it will take to complete all the epochs (in minutes).

        NOTE:
        - Deep Dream performs gradient ASCENT rather than DESCENT (which we are more used to). The difference is only
        in the sign of the gradients.
        - Clipping is different than normalization!
        '''
        self.loss_history = []
       
        for epoch in range(n_epochs):
            
            initial_time = time.time()
            loss, grads = self.forward(gen_img)
            self.loss_history.append(loss)
            gen_img.assign_add(grads*lr)
            gen_img.assign(tf.clip_by_value(gen_img, clip_value_min=0, clip_value_max=1))
            if plot:
                if (epoch) % print_every == 0:
                    image = tf_util.tf2image(gen_img)
                    fig = plt.figure(figsize=plot_fig_sz)
                    plt.imshow(image)
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()
                    #if export:
                        #image.save()
            print("epoch " + str(epoch +1)+' / '+str(n_epochs)+ ' completed')
            if epoch == 0:
                    elapsed_time = time.time() - initial_time
                    print("1 epoch took: "+str(elapsed_time) +" seconds." + "Expected runtime is:" +str(n_epochs * elapsed_time/60) + " minutes")
                
               
                
                
        return self.loss_history
            

        

    def fit_multiscale(self, gen_img, n_scales=4, scale_factor=1.3, n_epochs=26, lr=0.01, print_every=1, plot=True,
                       plot_fig_sz=(5, 5), export=True):
        '''Run DeepDream `fit` on the generated image `gen_img` a total of `n_scales` times. After each time, scale the
        width and height of the generated image by a factor of `scale_factor` (round to nearest whole number of pixels).
        The generated image does NOT start out from scratch / the original image after each resizing. Any modifications
        DO carry over across runs.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_scales: int. Number of times the generated image should be resized and DeepDream should be run.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current scale) every this many SCALES (not epochs).
        plot: bool. If true, plot/show the generated image `print_every` SCALES.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` SCALES. Each exported image should have the current scale number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Call fit `n_scale` times. Pass along hyperparameters (n_epochs, etc.). Turn OFF plotting and exporting within
        the `fit` method — this method should take over the plotting and exporting (in scale intervals rather than epochs).
        2. Multiplicatively scale the generated image.
            Check out: https://www.tensorflow.org/api_docs/python/tf/image/resize

            NOTE: The output of the built-in resizing process is NOT a tf.Variable (its an ordinary tensor).
            But we need a tf.Variable to compute the image gradient during gradient ascent.
            So, wrap the resized image in a tf.Variable.
        3. After the first scale completes, always print out how long it took to finish the first scale and an estimate
        of how long it will take to complete all the scales (in minutes).
        '''
        pass
