'''network.py
Represents  a neural network (collection of layers)
YOUR NAMES HERE
CS343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import time
import numpy as np

import layer
import filter_ops
import accelerated_layer


class Network:
    '''Represents a neural network with some number of layers of various types.
    To create a specific network, create a subclass (e.g. ConvNet4) then
    add layers to it. For this project, the focus will be on building the
    ConvNet4 network.
    '''
    def __init__(self, reg=0, verbose=True):
        '''This method is pre-filled for you (shouldn't require modification).
        '''
        # Python list of Layer object references that make up out network
        self.layers = []
        # Regularization strength
        self.reg = reg
        # Whether we want network-related debug/info print outs
        self.verbose = verbose

        # Python list of ints. These are the indices of layers in `self.layers`
        # that have network weights. This should be all types of layers
        # EXCEPT MaxPool2D
        self.wt_layer_inds = []

        # As in former projects, Python list of loss, training/validation
        # accuracy during training recorded at some frequency (e.g. every epoch)
        self.loss_history = []
        self.train_acc_history = []
        self.validation_acc_history = []

    def compile(self, optimizer_name, **kwargs):
        '''Tells each network layer how weights should be updated during backprop
        during training (e.g. stochastic gradient descent, adam, etc.)

        This method is pre-filled for you (shouldn't require modification).

        NOTE: This NEEDS to be called AFTER creating your ConvNet4 object,
        but BEFORE you call `fit()` to train the net (otherwise, how does your
        net know how to update the weights?).

        Parameters:
        -----------
        optimizer_name: string. Name of optimizer class to use to update wts.
            See optimizer::create_optimizer for specific ones supported.
        **kwargs: Any number of optional parameters that get passed to the
            optimizer of your choice. e.g. learning rate.
        '''
        # Only set an optimizer for each layer with weights
        for l in [self.layers[i] for i in self.wt_layer_inds]:
            l.compile(optimizer_name, **kwargs)

    def fit(self, x_train, y_train, x_validate, y_validate, mini_batch_sz=100, n_epochs=10, acc_freq=100, print_every=50):
        '''Trains the neural network on data

        Parameters:
        -----------
        x_train: ndarray. shape=(num training samples, n_chans, img_y, img_x).
            Training data.
        y_train: ndarray. shape=(num training samples,).
            Training data classes, int coded.
        x_validate: ndarray. shape=(num validation samples, n_chans, img_y, img_x).
            Every so often during training (see acc_freq param), we compute
            the accuracy of the network in classifying the validation set
            (out-of-training-set generalization). This is the data we use.
        y_validate: ndarray. shape=(num validation samples,).
            Validation data classes, int coded.
        mini_batch_sz: int. Mini-batch training size.
        n_epochs: int. Number of training epochs.
        print_every: int.
            Controls the frequency (in iterations) with which to wait before printing out the loss
            and iteration number.
            NOTE: Previously, you used number of epochs rather than iterations to measure the frequency
            of print-outs. Use the simpler-to-implement units of iterations here because CNNs are
            more computationally intensive and you may want print-outs during an epoch.
        acc_freq: int. Should be equal to or a multiple of `print_every`.
            How many training iterations (weight updates) we wait before computing accuracy on the
            full training and validation sets?
            NOTE: This is is a computationally intensive process for the big network so make sure
            that you only COMPUTE training and validation accuracies this often
            (i.e DON'T compute them every iteration).

        TODO: Complete this method's implementation.
        1. In the main training loop, randomly sample to get a mini-batch.
        2. Do forward pass through network using the mini-batch.
        3. Do backward pass through network using the mini-batch.
        4. Compute the loss on the mini-batch, add it to our loss history list
        5. Call each layer's update wt method.
        6. Add support for `print_every` and `acc_freq`.
        7. Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
            Also printout the projected time for completing ALL training iterations.
            (For simplicity, you don't need to consider the time taken for computing
            train and validation accuracy).

        '''
        iterations = 0
        num_batch_loops = (x_train.shape[0]//mini_batch_sz)

        for i in range(n_epochs):
            for batch in range(num_batch_loops):

                initial_time = time.time()
                #get batch: 
                samples = np.random.randint(0,x_train.shape[0], mini_batch_sz)
                batch_X = x_train[samples]
                batch_y = y_train[samples]

                #now we pass batch through forward:

                self.loss_history.append(self.forward(batch_X, batch_y))
                self.backward(batch_y)
                iterations +=1

                for layer in self.layers:
                    layer.update_weights()

                if iterations == 1:
                    elapsed_time = time.time() - initial_time
                    print("time for 0th iteration: "+str(elapsed_time) +" seconds")
                    print("projected time to finish: "+str(n_epochs * num_batch_loops * elapsed_time/60) + " minutes")
                
                if (iterations-1) % print_every == 0:
                    print("iterations number: "+str(iterations-1)+" ------- loss: ", self.loss_history[-1])
                
                if (iterations-1) % acc_freq == 0:
                    self.train_acc_history.append(self.accuracy(x_train, y_train))
                    self.validation_acc_history.append(self.accuracy(x_validate, y_validate))
                    print("train accuracy: ", self.train_acc_history[-1])
                    print("validation accuracy: ", self.validation_acc_history[-1])
                
        print("final loss: ", self.loss_history[-1])
        print("final train accuracy ", self.accuracy(x_train, y_train))
        
        

        pass

    def predict(self, inputs):
        '''Classifies novel inputs presented to the network using the current
        weights.

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num test samples, n_chans, img_y, img_x)
            This is test data.

        Returns:
        -----------
        pred_classes: ndarray. shape=shape=(num test samples)
            Predicted classes (int coded) derived from the network.
        '''

        # Do a forward sweep through the network
        prev_ins = inputs

        for l in self.layers:
            prev_ins = l.forward(prev_ins)
        #now we want to go to our output layer activations 
        #and see what classes they are for

        output = self.layers[-1]
        activation = output.net_act
        pred_classes = np.argmax(activation, axis = 1)
        return pred_classes


    def accuracy(self, inputs, y, samp_sz=500, mini_batch_sz=15):
        '''Computes accuracy using current net on the inputs `inputs` with classes `y`.

        This method is pre-filled for you (shouldn't require modification).

        Parameters:
        -----------
        inputs: ndarray. shape=shape=(num samples, n_chans, img_y, img_x)
            We are testing the classification accuracy on these data.
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(N,) for mini-batch size N.
        samp_sz: int. If the number of samples is bigger than this number,
            we take a random sample from `inputs` of this size. We do this to
            keep performance of this method reasonable.
        mini_batch_sz: Because it might be tricky to hold all the training
            instances in memory at once, process and evaluate the accuracy of
            samples from `input` in mini-batches. We merge the accuracy scores
            across batches so the result is no different than processing all at
            once.
        '''
        n_samps = len(inputs)

        # Do we subsample the input?
        if n_samps > samp_sz:
            subsamp_inds = np.random.choice(n_samps, samp_sz)
            n_samps = samp_sz
            inputs = inputs[subsamp_inds]
            y = y[subsamp_inds]

        # How many mini-batches do we split the data into to test accuracy?
        n_batches = int(np.ceil(n_samps / mini_batch_sz))
        # Placeholder for our predicted class ints
        y_pred = np.zeros(len(inputs), dtype=np.int32)

        # Compute the accuracy through the `predict` method in batches.
        # Strategy is to use a 1D cursor `b` to extract a range of inputs to
        # process (a mini-batch)
        for b in range(n_batches):
            low = b*mini_batch_sz
            high = b*mini_batch_sz+mini_batch_sz
            # Tolerate out-of-bounds as we reach the end of the num samples
            if high > n_samps:
                high = n_samps

            # Get the network predicted classes and put them in the right place
            y_pred[low:high] = self.predict(inputs[low:high])

        # Accuracy is the average proportion that the prediction matchs the true
        # int class
        acc = np.mean(y_pred == y)

        return acc

    def forward(self, inputs, y):
        '''Do forward pass through whole network

        Parameters:
        -----------
        inputs: ndarray. Inputs coming into the input layer of the net. shape=(B, n_chans, img_y, img_x)
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        loss: float. REGULARIZED loss.

        TODO:
        1. Call the forward method of each layer in the network.
            Make the output of the previous layer the input to the next.
        2. Compute and get the loss from the LAST network layer.
        2. Compute and get the weight regularization via `self.wt_reg_reduce()` (implement this next)
        4. Return the sum of the loss and the regularization term.
        '''
        
        prev_ins = inputs
        for l in self.layers:
            prev_ins = l.forward(prev_ins)
        loss = self.layers[-1].cross_entropy(y)
        wt_reg = self.wt_reg_reduce()
        return loss + wt_reg
        

    def wt_reg_reduce(self):
        '''Computes the loss weight regularization for all network layers that have weights

        Returns:
        -----------
        wt_reg: float. Regularization for weights from all layers across the network.

        NOTE: You only can compute regularization for layers with wts!
        Layer indicies with weights are maintained in `self.wt_layer_inds`.
        The network regularization `wt_reg` is simply the sum of all the regularization terms
        for each individual layer.
        '''
        
        wt_reg = 0
        for i in self.wt_layer_inds:
            wt_reg += 0.5 * self.reg * np.sum(np.square(self.layers[i].wts))
        return wt_reg

    def backward(self, y):
        '''Initiates the backward pass through all the layers of the network.

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
            shape=(B,) for mini-batch size B.

        Returns:
        -----------
        None

        TODO:
        1. Initialize d_upstream, d_wts, d_b to None.
        2. Loop through the network layers in REVERSE ORDER, calling the `Layer` backward method.
            Remember that the output of layer.backward() becomes the d_upstream to the next layer down.
            We don't care about d_wts, d_b in this method (computed/stored in Layer).
        '''
        pass

        d_upstream = None
        d_wts = None
        d_b = None
            
        for layer in reversed(self.layers) :
            d_upstream, d_wts, d_b = layer.backward(d_upstream, y)




class ConvNet4(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer
        first_layer = layer.Conv2D(0, "Conv", n_kers[0], ker_sz[0], n_chans, wt_scale, "relu", reg, False)

        # 2) 2x2 max pooling layer
        second_layer = layer.MaxPooling2D(1, "Pool", pooling_sizes[0], pooling_strides[0], "linear", reg, False)

        # 3) Dense layer
        x = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        
        max_out_shape = x*y*n_kers[0]
        third_layer = layer.Dense(2, "Dense", dense_interior_units[0], max_out_shape, wt_scale, "relu", reg, False)

        # 4) Dense softmax output layer
        
        fourth_layer = layer.Dense(3, "Last_Dense", n_classes, dense_interior_units[0], wt_scale, "softmax", reg, False)
        
        self.layers.append(first_layer)
        self.layers.append(second_layer)
        self.layers.append(third_layer)
        self.layers.append(fourth_layer)
        self.wt_layer_inds = [0,2,3]

        # self.wt_layer_inds = ???

class ConvNet4Accel(Network):
    '''
    Makes a ConvNet4 network with the following layers: Conv2D -> MaxPooling2D -> Dense -> Dense

    1. Convolution (net-in), Relu (net-act).
    2. Max pool 2D (net-in), linear (net-act).
    3. Dense (net-in), Relu (net-act).
    4. Dense (net-in), soft-max (net-act).
    '''
    def __init__(self, input_shape=(3, 32, 32), n_kers=(32,), ker_sz=(7,), dense_interior_units=(100,),
                 pooling_sizes=(2,), pooling_strides=(2,), n_classes=10, wt_scale=1e-3, reg=0.001, verbose=True):
        '''
        Parameters:
        -----------
        input_shape: tuple. Shape of a SINGLE input sample (no mini-batch). By default: (n_chans, img_y, img_x)
        n_kers: tuple. Number of kernels/units in the 1st convolution layer. Format is (32,), which is a tuple
            rather than just an int. The reasoning is that if you wanted to create another Conv2D layer, say with 16
            units, n_kers would then be (32, 16). Thus, this format easily allows us to make the net deeper.
        ker_sz: tuple. x/y size of each convolution filter. Format is (7,), which means make 7x7 filters in the FIRST
            Conv2D layer. If we had another Conv2D layer with filters size 5x5, it would be ker_sz=(7,5)
        dense_interior_units: tuple. Number of hidden units in each dense layer. Same format as above.
            NOTE: Does NOT include the output layer, which has # units = # classes.
        pooling_sizes: tuple. Pooling extent in the i-th MaxPooling2D layer.  Same format as above.
        pooling_strides: tuple. Pooling stride in the i-th MaxPooling2D layer.  Same format as above.
        n_classes: int. Number of classes in the input. This will become the number of units in the Output Dense layer.
        wt_scale: float. Global weight scaling to use for all layers with weights
        reg: float. Regularization strength
        verbose: bool. Do we want to term network-related debug print outs on?
            NOTE: This is different than per-layer verbose settings, which are turned manually on below.

        TODO:
        1. Assemble the layers of the network and add them (in order) to `self.layers`.
        2. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
        '''
        super().__init__(reg, verbose)

        n_chans, h, w = input_shape

        # 1) Input convolutional layer
        first_layer = accelerated_layer.Conv2DAccel(0, "Conv", n_kers[0], ker_sz[0], n_chans, wt_scale, "relu", reg, False)

        # 2) 2x2 max pooling layer
        second_layer = accelerated_layer.MaxPooling2DAccel(1, "Pool", pooling_sizes[0], pooling_strides[0], "linear", reg, False)

        # 3) Dense layer
        x = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        y = filter_ops.get_pooling_out_shape(h, pooling_sizes[0], pooling_strides[0])
        
        max_out_shape = x*y*n_kers[0]
        third_layer = layer.Dense(2, "Dense", dense_interior_units[0], max_out_shape, wt_scale, "relu", reg, False)

        # 4) Dense softmax output layer
        
        fourth_layer = layer.Dense(3, "Last_Dense", n_classes, dense_interior_units[0], wt_scale, "softmax", reg, False)
        
        self.layers.append(first_layer)
        self.layers.append(second_layer)
        self.layers.append(third_layer)
        self.layers.append(fourth_layer)
        self.wt_layer_inds = [0,2,3]

        # self.wt_layer_inds = ???


class EncoderDecoderSmall(Network):
    #Makes a single encoding-decoding network with the following layers:

    #1. Input (16,16,1)
    #2. Conv2dAccel (16,16,16) with 16 3x3 kernals
    #3. Conv2dAccel (16,16,16) with 16 3x3 kernals
    #4. max-poolingAccel 2x2 (8,8,16) 
    #5. Conv2dAccel (8,8,32) with 32 3x3 kernals
    #6. Conv2dAccel (8,8,32) with 32 3x3 kernals
    #7. transposed convolution? (16,16,16)
    #8. ---- Copy and Concat from step 3. -----> (16,16,32)
    #9. Conv2dAccel (16,16,16) with 16 3x3 kernals ____ Do we want to use more kernals
    #10. Conv2dAccel (16,16,16) with 16 3x3 kernals
    #11. Dense (16,16,1) actually a conv2d with a 1x1 kernal ? 

    def __init__(self, wt_scale=1e-3, reg=0.001, verbose=True):
        
        super().__init__(reg, verbose)
    
        #1. Input:
        n_chans = 1
        n_classes = 3
        n_kers = (16,32,16,2)
        ker_sz = (3,)
        pooling_sizes = (2,)
        pooling_strides = (2,) #could be 1 not sure.


        #2. Conv2d
        second_layer = layer.Conv2D(0, "Conv", n_kers[0], ker_sz[0], n_chans, wt_scale, "relu", reg, verbose)

        #3. Conv2d
        third_layer = layer.Conv2D(1, "Conv", n_kers[0], ker_sz[0], n_kers[0], wt_scale, "relu", reg, verbose)

        #4. max-pooling
        fourth_layer = layer.MaxPooling2D(2, "Pool", pooling_sizes[0], pooling_sizes[0], "linear", reg, verbose)

        #5. Conv2d

        fifth_layer = layer.Conv2D(3, "Conv", n_kers[1], ker_sz[0],n_kers[0], wt_scale, "relu", reg, verbose)

        #6. Conv2d

        sixth_layer = layer.Conv2D(4, "Conv", n_kers[1], ker_sz[0], n_kers[1], wt_scale, "relu", reg, verbose)

        #7. Upconvolution

        seventh_layer = layer.UpConvolution(5, "UpConv",n_kers[1], 2, n_kers[1], wt_scale, 'linear', reg, verbose) #not sure about the number of kernals here

        
        #8. CopyAndConcat

        #eighth_layer = layer.CopyAndConcat(6,"CopyAndConcat",verbose = verbose) #concats across channel dimension so we will have n_kers_in*2

        #9. Conv2d

        ninth_layer = layer.Conv2D(6, "Conv", n_kers[2], ker_sz[0], n_kers[1], wt_scale, "relu", reg, verbose)

        #10. Conv2d

        tenth_layer =  layer.Conv2D(7, "Conv", n_kers[2], ker_sz[0], n_kers[2], wt_scale, "relu", reg, verbose)

        #11. Dense Maybe add a second dense
        size = 16*16*16
        eleventh_layer =  layer.Dense(8, "Dense",n_classes, size, wt_scale, "softmax",reg, verbose)

   
        self.layers.append(second_layer)
        self.layers.append(third_layer)
        self.layers.append(fourth_layer)
        self.layers.append(fifth_layer)
        self.layers.append(sixth_layer)
        self.layers.append(seventh_layer)
       # self.layers.append(eighth_layer)
        self.layers.append(ninth_layer)
        self.layers.append(tenth_layer)
        self.layers.append(eleventh_layer)
        self.wt_layer_inds = [0,1,3,4,5,6,7,8] 









