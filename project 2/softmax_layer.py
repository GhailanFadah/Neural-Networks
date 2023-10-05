'''softmax_layer.py
Constructs, trains, tests single layer neural network with softmax activation function.
YOUR NAMES HERE
CS 343: Neural Networks
Fall 2023
Project 2: Multilayer Perceptrons
'''
import numpy as np


class SoftmaxLayer:
    '''
    SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''

        return np.sum(y == y_pred)/y.size

    def net_in(self, features):
        '''Computes the net input (net weighted sum)
        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''

        net_in = features @ self.wts + self.b
        return net_in
    
    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''

        y_one_hot = np.zeros((y.shape[0], num_classes))
        i = 0
        for int in y:  
            y_one_hot[i,int] = 1
            i+=1
        
        return y_one_hot


    def fit(self, features, y, n_epochs=10000, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and
            training loss every 100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.

        NOTE:
        Recall: training epoch is not the same thing as training iteration with mini-batch.
        If we have mini_batch_sz = 100 and N = 1000, then we have 10 iterations per epoch. Epoch
        still means entire pass through the training data "on average". Print this information out
        if verbose > 0.

        TODO:
        -----------
        1) Initialize the wts/bias to small Gaussian numbers:
            mean 0, std 0.01, Wts shape=(num_feat M, num_classes C), b shape=(num_classes C,)
        2) Implement mini-batch support: On every iter draw from our input samples (with replacement)
        a batch of samples equal in size to `mini_batch_sz`. Also keep track of the associated labels.
        THEY MUST MATCH UP!!
            - Keep in mind that mini-batch wt updates are different than epochs. There is a parameter
              for E (num epochs), not number of iterations.
            - Handle this edge case: we do SGD and mini_batch_sz = 1. Add a singleton dimension
              so that the "N"/sample_size dimension is still defined.
        4) Our labels are int coded (0,1,2,3...) but this representation doesnt work well for piping
        signals to the C output neurons (C = num classes). Transform the mini-batch labels to one-hot
        coding from int coding (see function above to write this code).
        5) Compute the "net in".
        6) Compute the activation values for the output neurons (you can defer the actual function
        implementation of this for later).
        7) Compute the cross-entropy loss (again, you can defer the details for now)
        8) Do backprop:
            a) Compute the error gradient for the mini-batch sample,
            b) update weights using gradient descent.

        HINTS:
        -----------
        2) Work in indices, not data elements.
        '''

        num_samps, num_features = features.shape
        num_classes = self.num_output_units

        self.b = np.random.normal(0, 0.01, num_classes)

        self.wts = np.reshape(np.random.normal(0, 0.01, num_samps*num_features), (num_samps, num_features))
        
        for epoch in range(n_epochs):
            for batch_n in range(int((num_samps)/mini_batch_sz)):
                
                batch_inds = np.random.randint(0,num_classes,mini_batch_sz)
                mb_X = features[batch_inds]
                mb_y = self.one_hot(y[batch_inds])

                #HANDLE EDGE CASE FOR BATCH SIZE = 1
                net_in = self.net_in(features)
                net_act = self.activation(net_in)
                loss = self.loss(net_act,y, reg)

                #BACKPROP
                gradient_b, gradient_w = self.gradient(features, net_act, y, reg)
                
                w_step = lr*gradient_w 
                b_step = lr*gradient_b 
            
                self.wts = self.get_wts()-w_step 
                self.b = self.get_bias()-b_step

        pass

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        
        
        pass

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        z_max = np.max(net_in, axis = 1, keepdims=True)
        adj_net = net_in - z_max

        net_act = np.exp(adj_net)/np.sum(np.exp(adj_net), axis=1, keepdims=True)
        return net_act

    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to select only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        '''
        
        
        print(y)
        #get array of net_act[y]
        corrects = net_act[np.arange(net_act.shape[0]), y]
        
        #cross-entropy loss with bias term from 9/20 class
        loss = -(1/y.size)*np.sum(np.log(corrects))+(1/2)*reg*np.sum(self.wts**2)
        return loss
    
        

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''
        errors = net_act - y
        mini_batch_size = net_act.shape[0]

        return (-1/mini_batch_size)*np.sum(errors), (-1/mini_batch_size)*(errors.T @ features)

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.6770080064951229, max=1.4469478863251897\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.0688478818744139, max=0.15020377627180137\n')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are:\n{net_in[0, :5]}\nand should be')
        print('[0.864 0.927 0.874 1.159 1.392]')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are:\n{net_act[0, :5]}\nand should be')
        print('[0.078 0.083 0.079 0.105 0.132]')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}.')
        print('You should know what the sum should be :)')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
