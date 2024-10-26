'''contastive_hebb_net.py
Hebbian neural network that learns using the contrastive hebbian learning rule
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import tensorflow as tf

class Layer:
    '''One MLP-like layer of the Contrastive Hebbian Network

    NOTE: Should only use TensorFlow (no NumPy). Your challenge is to NOT import numpy in this file!
    '''
    def __init__(self, num_neurons, num_neurons_below, layer_above=None, wt_scale=0.6):
        '''Layer constructor

        Parameters:
        -----------
        num_neurons: int. Number of neurons/units in the current layer (H).
        num_neurons_below: int. Number of neurons/units in the layer below the current layer (M) (i.e. closer to the input
            layer).
        layer_above: Layer or None. Object representing the layer above the current one in the network (if applicable).
            Could be None if there is no layer above (i.e. current layer is output layer) or if we are setting the layer
            above layer via `set_layer_above` method.
        wt_scale: float. Maximum absolute value of any weight or bias belonging to the current layer (i.e. wts from the
            layer below and current layer).

        TODO:
        - Set instance variables as needed.
        - Instantiate weights and bias according to a uniform distribition with values from -`wt_scale` to +`wt_scale`.
        '''

        self.curr_net_act = None  # Keep me to represent the last computed net_act
        self.num_neurons = num_neurons
        self.num_neurons_below = num_neurons_below
        self.layer_above = layer_above
        
        self.wts = tf.Variable(tf.random.uniform((num_neurons_below, num_neurons), -wt_scale, wt_scale))
        self.b = tf.Variable(tf.random.uniform((num_neurons,1),-wt_scale, wt_scale))

    def get_num_neurons(self):
        '''Returns the number of units in the current layer.'''
        
        return self.num_neurons
        

    def get_wts(self):
        '''Returns the weights in the current layer. shape=(M, H), where `M` is the number of neurons in the layer below
        and `H` is the number of neurons in the current layer.'''
        
        return self.wts
        

    def get_b(self):
        '''Returns the bias in the current layer. shape=(H,), where`H` is the number of neurons in the current layer.'''
        
        return self.b
        
    
    def get_layer_above(self):
        '''Returns the `Layer` object representing the layer above the current one.'''
        
        return self.layer_above
        

    def get_curr_net_act(self):
        '''Returns the last computed net_act. Could be None if this was never computed before.'''
        
        return self.curr_net_act

    def set_curr_net_act(self, curr_net_act):
        '''Replace the last computed net_act with `curr_net_act`.'''
        
        self.curr_net_act = curr_net_act

    def set_layer_above(self, layer_above):
        '''Sets the `Layer` object representing the layer above to `layer_above`.'''
        
        self.layer_above = layer_above
    
    def reset_state(self, N):
        '''Resets the layer state (last computed net_act) to the default value, a (N, H) tensor of 0s.

        Parameters:
        -----------
        N: int. Number of samples in the current mini-batch.
        '''
        reset = tf.zeros((N, self.get_num_neurons()))
        self.set_curr_net_act(reset)

    def net_in(self, x, gamma):
        '''Computes the net input for every neuron in the current layer.

        Parameters:
        -----------
        x: tf.constant. shape=(N, M). Input to the current layer from the layer below (or input layer if there is no
            hidden layer below). N is the number of samples in the mini-batch and M is the number of neurons in the layer
            below.
        gamma: float. Feedback strength hyperparameter controlling the magnitude of the feedback from the layer above
            (if there is a layer above).

        HINT: This is just like MLP net_in but there is the feedback signal. This should only be applied IF there is a
        layer above (see refresher of equation in notebook).
        '''
        first_part = x @ self.get_wts() + tf.transpose(self.get_b())
        layer_above = self.get_layer_above() 

        if layer_above == None:
            return first_part
        wts_above = layer_above.get_wts()
        above_net_act = layer_above.get_curr_net_act()
        if above_net_act == None or tf.shape(above_net_act)[0] != tf.shape(x)[0]:
            return first_part
        else:
            second_part = gamma * (above_net_act @ tf.transpose(wts_above))
            net_in = first_part + second_part
            return net_in
        
        

    def net_act(self, net_input):
        '''Computes the net activation (sigmoid activation function) for every neuron in the current layer.

        Parameters:
        -----------
        net_input: tf.constant. shape=(N, H). Net input of the current layer.
        gamma: float. Feedback strength hyperparameter controlling the magnitude of the feedback from the layer above
            (if there is a layer above).

        NOTE: Don't forget to set the current net_act instance variable.
        '''
        #net_act = tf.keras.activations.sigmoid(net_input)
        net_act = 1/(1+tf.math.exp(-net_input))
        self.set_curr_net_act(net_act)
        return net_act

    def update_wts(self, d_wts, d_b, lr):
        '''Updates the weight and bias based on the weight and bias changes passed in.

        Applies the update:
        wts(t+1) = wts(t) + lr*d_wts
        b(t+1) = b(t) + lr*d_b

        Parameters:
        -----------
        d_wts: tf.constant. shape=(M, H). Amount by which we should update each weight.
        d_b: tf.constant. shape=(H,). Amount by which we should update each bias.
        lr: float. Learning rate.
        '''

        self.wts = self.wts+lr*d_wts
        self.b = self.b+lr*d_b
        pass

class ContrastiveNet:
    '''MLP-like network composed of `L` layers that learns using the Contrastive Hebbian rule.
    '''
    def __init__(self, num_neurons_each_layer, wt_scale=0.6, gamma=0.01, n_iter=10):
        '''ContrastiveNet constructor

        Parameters:
        -----------
        num_neurons_each_layer: tuple of ints. Number of neurons in each layer of the network.
            INCLUDES the number of neurons/features in the input layer.
        wt_scale: float. Maximum absolute value of any weight or bias belonging to layers in the network.
        gamma: float. Feedback strength hyperparameter controlling the magnitude of the feedback from each layer to the
            layer below).
        n_iter: int. Number of iterations that should be used in the Free and Clamped phases to compute the activation
            in each layer.

        TODO:
        - Set instance variables as needed.
        - Create a list of all the `Layer` objects in the network. Only make `Layer` objects for NON-INPUT layers.
            - Store them in ascending order in a list (set as instance variable).
            - You can build this list out with a single loop, but it may be easier to think about using two successive loops:
                1. Create the layers without layers above defined.
                2. Associate the layer above with the current layer.
            - Be careful with loop indices and off-by-one errors!!!
        '''
        self.layers = []
        self.gamma = gamma
        self.n_iter = n_iter
        #create layers without above guys
        for i,layer_sz in enumerate(num_neurons_each_layer):
            if i == 0: 
                continue
            
            new_layer = Layer(layer_sz,num_neurons_each_layer[i-1], wt_scale = wt_scale)
            self.layers.append(new_layer)
            
        for i,layer in enumerate(self.layers): 
            if i == len(self.layers)-1:
                continue
            layer.layer_above = self.layers[i+1]

        pass

    def get_layers(self):
        '''Returns the list of `L` `Layer` objects that make up the network.'''
        return self.layers

    def get_net_acts_all_layers(self, x=None):
        '''Gets a list of the current net_acts in each layer of the network.

        Parameters:
        -----------
        x: tf.constant or `None`. shape=(N, M). The mini-batch at the input layer.
            If None, return the net_acts of all `L` non-input layers of the network.
            If not None, include the mini-batch input `x` as the first item of the list so that it has length `L+1`.

        Returns:
        -----------
        Python list. len=`L` or `L+1`. The current net_acts in each layer of the network.
        '''
        net_acts_all = []
        if x is not None:
            net_acts_all.append(x)
        for layer in self.layers:
            net_acts_all.append(layer.curr_net_act)
        return net_acts_all
    
    def set_gamma(self, gamma):
        '''Set the feedback strength hyperparameter to the passed in value.'''
        self.gamma = gamma
    
    def set_n_iter(self, n_iter):
        '''Set the number of Free and Clamped Phase iterations to the passed in value.'''
        self.n_iter = n_iter

    def one_hot(self, y, C):
        '''One-hot codes the vector of class labels `y`

        Parameters:
        -----------
        y: tf.constant. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.
            e.g. if y=[1, 0], and C=3, the one-hot vector would be:
            [[0., 1., 0.], [1., 0., 0.]]
        '''
        return tf.one_hot(y,depth = C)

    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y_true: tf.constant. shape=(B,). int-coded true classes.
        y_pred: tf.constant. shape=(B,). int-coded predicted classes by the network.

        Returns:
        -----------
        float. accuracy in range [0, 1]

        Hint: tf.where might be helpful.
        '''
        correct = tf.where(y_true == y_pred)
        correct_num = correct.shape[0]
        
        
        return correct_num/y_true.shape[0]

    def free_step(self, x):
        '''Do one step of the Free Phase. This consists of completing `n_iter` "forward passes" through the network on
        which each layer's activation is computed using the current mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. shape=(N, M). The mini-batch at the input layer.

        NOTE:
        - Don't forget to 0 out any existing net act values in the network layers before starting otherwise state from
        a previously completed phase could unintentionally influence your results!
        - The current mini-batch input should remain present/"on" at the input layer throughout the entire Free Phase step.
        - You will need to think about how to handle feeding the first hidden layer its input.
        '''

        next_input = x
        for i in range(self.n_iter):
            for layer in self.layers:
                if i==0:
                    layer.reset_state(tf.shape(next_input)[0])
                
                net_in = layer.net_in(next_input, gamma = self.gamma)
                net_act = layer.net_act(net_in)
                #print(net_act)
                next_input = net_act
                if layer.layer_above == None:
                    next_input = x
            

    def clamped_step(self, x, yh):
        '''Do one step of the Clamped Phase. This is identical to the Free Phase except:

        1. We fix (*clamp*) the output layer to the one-hot coded true classes when processing the current mini-batch.
        2. We do NOT modify/touch these fixed output layer activations when doing each "forward pass"!

        Parameters:
        -----------
        x: tf.constant. shape=(N, M). The mini-batch at the input layer.
        yh: tf.constant. shape=(N, C). The one-hot coding of the mini-batch sample labels.

        NOTE: Don't forget to 0 out any existing net act values in the network layers before starting otherwise state from
        a previously completed phase could unintentionally influence your results!

        NOTE: You are encouraged to copy-paste from your `free_step` implementation!
        '''
        next_input = x
        for i in range(self.n_iter):
            for layer in self.layers:
                if not i:
                    layer.reset_state(tf.shape(x)[0])
                
                net_in = layer.net_in(next_input, gamma = self.gamma)
                net_act = layer.net_act(net_in)
              
                next_input = net_act
                if layer.layer_above == None:
                    next_input = x
                    layer.curr_net_act = yh

        pass

    def update_wts(self, free_acts_all_layers, clamped_acts_all_layers, lr):
        '''Updates the wts and bias in each network layer using the Contrastive Hebbian Learning Rule (see notebook for
        refresher).

        Parameters:
        -----------
        free_acts_all_layers: Python list. len=L+1. The final net_act values in each of the `L` non-input network layers
            after the Free Phase is complete for the current mini-batch. First item in list is the current mini-batch
            input in the input layer.
        clamped_acts_all_layers: Python list. len=L+1. The final net_act values in each of the `L` non-input network layers
            after the Clamped Phase is complete for the current mini-batch. First item in list is the current mini-batch
            input in the input layer.
        lr: float. The learning rate.
        '''
        B = free_acts_all_layers[0].shape[0]


        for layer in range(len(free_acts_all_layers)):
            if layer == 0:
                continue
            #layer -1 gives the weights from the description
            s = (1/B)*tf.math.pow(self.gamma,(layer)-(len(self.layers)))
            
            clamped_act_below_T = tf.transpose(clamped_acts_all_layers[layer-1])
            clamped_act = clamped_acts_all_layers[layer]

            free_act_below_T = tf.transpose(free_acts_all_layers[layer-1])
            free_act = free_acts_all_layers[layer]

            d_wts_layer = s * (clamped_act_below_T @ clamped_act-free_act_below_T @ free_act)

            d_b_layer = s * (1/B) * tf.reduce_sum(clamped_act-free_act)

            self.get_layers()[layer-1].update_wts(d_wts_layer, d_b_layer, lr)
        

    def predict(self, x):
        '''Predicts the classes associated with the input data samples `x`. Predictions should be based on the neurons
        that achieve the highest netActs after running a Free Phase to the current mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. shape=(N, M). The input data samples.

        Returns:
        -----------
        y_preds: tf.constant. shape=(N,). The int-coded labels predicted for each data sample.
        '''

        self.free_step(x)
        predicted_acts = self.layers[-1].get_curr_net_act()
        y_preds = tf.math.argmax(predicted_acts, axis = 1)

        return y_preds


    def fit(self, x, y, x_val, y_val, epochs=1, batch_size=1024, lr=0.5):
        '''Train the network in mini-batches for `epochs` epochs. Training loop consists of the Free Phase,
        Clamped Phase, and a weight update. 

        Parameters:
        -----------
        x: tf.constant. shape=(N, M). The training data samples.
        y: tf.constant. shape=(N,). The int-coded labels for each training sample.
        x_val: tf.constant. shape=(N, M). The validation data samples.
        y_val: tf.constant. shape=(N,). The int-coded labels for each validation sample.
        epochs: int. Number of epochs over which to train the network.
        batch_size: int. Size of mini-batches used during training.
        lr: float. Learning rate for weight/bias update.

        Returns:
        -----------
        Python list. len=`epochs`. Training accuracy computed after every epoch of training.
        Python list. len=`epochs`. Validation accuracy computed after every epoch of training.

        NOTE:
        1. This is a `fit` method is structured fairly normally.
        2. Don't forget that the Clamped Phase expects the training labels in one-hot coded form.
        3. You probably should shuffle samples across epochs (or sample with replacement).
        4. You should neatly print out the training progress after each epoch. This should include current epoch,
        current training accuracy, current validation accuracy.
        '''
        N = len(x)

        train_acc_hist = []
        val_acc_hist = []

        for epoch in range(epochs):
            for batch in range(int(N/batch_size)):
                
                indices = tf.random.uniform((batch_size,),0,N-1, dtype = 'int32')
                batch_x = tf.gather(x, indices)
                batch_y = tf.gather(y, indices)
                one_hot_y_batch = self.one_hot(batch_y, 10) 

                #free on batch_x
                self.free_step(batch_x)

                #retrieve free acts
                free_acts = self.get_net_acts_all_layers(batch_x)

                #clamped on batch_x
                self.clamped_step(batch_x, one_hot_y_batch)

                #retrieve clamped acts
                clamped_acts = self.get_net_acts_all_layers(batch_x)
                
                #now update weights
                self.update_wts(free_acts, clamped_acts, lr)
            
            #end of epoch so check training set and validation set performances
            train_acc = self.accuracy(y,self.predict(x))
            val_acc = self.accuracy(y_val,self.predict(x_val))
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)


            print(f'End of epoch {epoch+1}/{epochs}: Train accuracy is {train_acc:.4f}, Validation accuracy is {val_acc:.4f}')
        return train_acc_hist, val_acc_hist