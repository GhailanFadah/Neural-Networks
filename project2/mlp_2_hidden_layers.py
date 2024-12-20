'''mlp.py
Constructs, trains, tests 2 layer multilayer layer perceptron networks
YOUR NAMES HERE
CS 343: Neural Networks
Fall 2023
Project 2: Multilayer Perceptrons
'''
import numpy as np


class MLP_extension:
    '''MLP is a class for multilayer perceptron network.

    The structure of our MLP will be:

    Input layer (X units) ->
    Hidden layer (Y units) with Rectified Linear activation (ReLu) ->
    Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that the current input sample belongs
    to class i.
    '''
    def __init__(self, num_input_units, num_hidden_units_1, num_hidden_units_2, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units_1 = num_hidden_units_1
        self.num_output_units = num_output_units
        self.num_hidden_units_2 = num_hidden_units_2

        self.initialize_wts(num_input_units, num_hidden_units_1, num_hidden_units_2, num_output_units)

    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y1_wts.copy()

    def initialize_wts(self, M, H_1, H_2, C, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.y_wts, self.y_b and self.z_wts, self.z_b
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0. For consistency with the test code, use the following random seeds:
                0: before defining self.y_wts
                1: before defining self.y_b
                2: before defining self.z_wts
                3: before defining self.z_b
        For example, use np.random.seed(0) before defining self.y_wts, np.random.seed(1) before defining self.y_b, etc.
          - For wt shapes, they should be be equal to (#prev layer units, #associated layer units)
            for example: self.y_wts has shape (M, H)
          - For bias shapes, they should equal the number of units in the associated layer.
            for example: self.y_b has shape (H,)
        '''
        
        np.random.seed(0)
        self.y1_wts = np.reshape(np.random.normal(0, std, M*H_1),(M,H_1))
        np.random.seed(1)
        self.y1_b = np.reshape(np.random.normal(0, std, H_1),(H_1,))
        np.random.seed(2)
        self.z_wts = np.reshape(np.random.normal(0, std, H_2*C),(H_2,C))
        np.random.seed(3)
        self.z_b = np.reshape(np.random.normal(0, std, C),(C,))
        np.random.seed(4)
        self.y2_wts = np.reshape(np.random.normal(0, std, H_1*H_2),(H_1,H_2))
        np.random.seed(5)
        self.y2_b = np.reshape(np.random.normal(0, std, H_2),(H_2,))

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

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        
        y_one_hot = np.zeros((y.shape[0], num_classes))
        i = 0
        for ing in y.astype(int):  
            y_one_hot[i,ing] = 1
            i+=1
        
        return y_one_hot
      

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            NOTE: You can figure out the predicted class assignments without applying the
            softmax net activation function — it will not affect the most active neuron.
        '''
        y1_net_in = features @ self.y1_wts + self.y1_b
        y1_net_act = np.where(y1_net_in<0,0,y1_net_in)

        # hidden function: sigmoid
        y2_net_in = y1_net_act @ self.y2_wts + self.y2_b
        y2_net_act = 1/(1 + np.exp(-y2_net_in))
        
        z_net_in = y2_net_act @ self.z_wts + self.z_b 
        
        y_pred = np.argmax(z_net_in, axis=1)
        return y_pred
    
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
    
    def loss(self, net_act, y, wts_1, wts_2, wts_3, reg=0):
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
        
        
        #get array of net_act[y]
        corrects = net_act[np.arange(net_act.shape[0]), y.astype(int)]
        #cross-entropy loss with bias term from 9/20 class
        
        #THERE IS AN ISSUE WHEN WE GET NETACT = 0 SINCE WE CANT TAKE LOG(0)
        loss = -(1/y.size)*np.sum(np.log(corrects))+(1/2)*reg*(np.sum(wts_1**2) + np.sum(wts_2**2) + np.sum(wts_3**2))
        return loss

    def forward(self, features, y, reg=0):
        '''Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity to the output layer, ending with the cross-entropy
        loss computation.

        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all in one method is not good design,
        but as you will discover, having the forward computations (y_net_in, y_net_act, etc) easily accessible in one
        place makes the backward pass a lot easier to track during implementation. In future projects, we will rely on
        better OO design.

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y1_net_in: ndarray. shape=(N, H_1). hidden layer "net in"
        y1_net_act: ndarray. shape=(N, H_1). hidden layer activation
        y2_net_in: ndarray. shape=(N, H_2). hidden layer "net in"
        y2_net_act: ndarray. shape=(N, H_2). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        
        # hidden function: ReLu
        y1_net_in = features @ self.y1_wts + self.y1_b
        y1_net_act = np.where(y1_net_in<0,0,y1_net_in)

        # hidden function: sigmoid
        y2_net_in = y1_net_act @ self.y2_wts + self.y2_b
        
        
        y2_net_act = 1/(1 + np.exp(-y2_net_in))
        
        z_net_in = y2_net_act @ self.z_wts + self.z_b 
        z_net_act = self.activation(z_net_in)
        
        loss = self.loss(z_net_act, y, self.z_wts, self.y1_wts, self.y2_wts, reg)
        
        return y1_net_in, y1_net_act, y2_net_in, y2_net_act, z_net_in, z_net_act, loss
        
    

    def backward(self, features, y, y1_net_in, y1_net_act, y2_net_in, y2_net_act, z_net_in, z_net_act, reg=0):
        '''Performs a backward pass (output -> hidden -> input) during training to update the weights. This function
        implements the backpropogation algorithm.

        This should start with the loss and progate the activity backwards through the net to the input-hidden weights.

        I added dz_net_act for you to start with, which is your cross-entropy loss gradient.
        Next, tackle dz_net_in, dz_wts and so on.

        I suggest numbering your forward flow equations and process each for relevant gradients in reverse order until
        you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights! (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Regularize each layer's weights like usual.
        '''
        N = z_net_act.shape[0]
        C = z_net_act.shape[1]
        dz_net_act = - (1/z_net_act)* 1/N
   
        dz_net_in = dz_net_act*(z_net_act*(self.one_hot(y, z_net_act.shape[1]) - z_net_act))
        
        dz_wts = (dz_net_in.T @ y2_net_act).T
      
        dz_b = np.sum(dz_net_in, axis=0)
        
        dy2_net_act = dz_net_in @ (self.z_wts).T
        
    
        dy2_net_in = dy2_net_act * (y2_net_act*(1-y2_net_act))
        
        dy2_wts = (dy2_net_in.T @ y1_net_act).T  # look at shapes  
        dy2_b = np.sum(dy2_net_in, axis=0) 
        
        dy1_net_act = dy2_net_in @ self.y2_wts.T
        
        x = np.copy(dy1_net_act)
        
        x[y1_net_in <= 0] = 0
        x[y1_net_in > 0] = 1
        dy1_net_in = dy1_net_act * x 
    
        dy1_wts = (dy1_net_in.T @ features).T
        dy1_b = np.sum(dy1_net_in, axis=0)
        
        
        # add reg term
        dz_wts += reg * self.z_wts
        dy1_wts += reg * self.y1_wts
        dy2_wts += reg * self.y2_wts
        
        return dy1_wts, dy1_b, dy2_wts, dy2_b, dz_wts, dz_b
        
    

    def fit(self, features, y, x_validation, y_validation,
            resume_training=False, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2, print_every=100):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Changes from `fit` in `SoftmaxLayer`:
        -------------------------------------
        1. Record accuracy on the validation set (`x_validation`, `y_validation`) after each epoch training.
        2. Record accuracy on training set after each epoch training.

        (see note below for more details)

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        resume_training: bool.
            False: we clear the network weights and do fresh training
            True: we continue training based on the previous state of the network.
                This is handy if runs of training get interupted and you'd like to continue later.
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the model per training epoch
            before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            after completing epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats. len=`n_epochs * n_iter_per_epoch`.
            Recorded training loss for each mini-batch of training.
        train_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the training set.
        validation_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the validation set.

        NOTE:
        The flow of this method should follow the one that you wrote in `SoftmaxLayer`. The main differences are:
        0) Remember to update weights and biases for ALL layers!
        1) Record the accuracy:
            - on training set: Compute it on the ENTIRE training set only after completing an epoch.
            - on validation set: Compute it on the ENTIRE validation set only after completing an epoch.
        2) As in `SoftmaxLayer`, loss on training set should be recorded for each mini-batch of training.
        3) Every `print_every` epochs, print out (if `verbose` is `True`):
        '''
        num_samps, num_features = features.shape
        num_classes = self.num_output_units
        loss_history = []
        acc_train = []
        acc_val = []
        if resume_training != True:
            self.initialize_wts
            
        for epoch in range(n_epochs):
            for batch_n in range(int((num_samps)/mini_batch_sz)):
                
                batch_inds = np.random.randint(0,num_samps,mini_batch_sz)
                mb_X = features[batch_inds]
                mb_y = self.one_hot(y[batch_inds],num_classes)
                y1_net_in, y1_net_act, y2_net_in, y2_net_act, z_net_in, z_net_act, loss = self.forward(mb_X, y[batch_inds], reg)
                loss_history.append(loss)

                
                dy1_wts, dy1_b, dy2_wts, dy2_b, dz_wts, dz_b = self.backward(mb_X, y[batch_inds], y1_net_in, y1_net_act, y2_net_in, y2_net_act, z_net_in, z_net_act, reg)
                self.y1_wts -= (dy1_wts*lr)
                self.y1_b -= (dy1_b*lr)
                self.y2_wts -= (dy2_wts*lr)
                self.y2_b -= (dy2_b*lr)
                self.z_wts -= (dz_wts*lr)
                self.z_b -= (dz_b*lr)

            acc_train.append(self.accuracy(y, self.predict(features)))
            acc_val.append(self.accuracy(y_validation, self.predict( x_validation))) 

            if epoch % print_every == 0 and verbose > 0:
                print("epoch: "+str(epoch)+" ------ loss: "+str(loss_history[-1])+ " \n------ train acc:"+ str(acc_train[-1])+ " ------ val acc: "+ str(acc_val[-1])+"\n")


        return loss_history, acc_train, acc_val