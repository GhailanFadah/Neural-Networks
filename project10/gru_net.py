'''gru_net.py
Gated Recurrent Unit (GRU) neural network for learning and predicting sequences of text characters
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 5: Recurrent Neural Networks
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def one_hot(x, num_feats):
    '''One-hot codes `x` into vectors of length `num_feats`

    Parameters:
    -----------
    x: tf Tensor. shape=(B, T). B is the mini-batch size and T is the sequence length.
    num_feats: int. Desired length of each one-hot vector.
    
    Returns:
    -----------
    one-hot coded input: tf Tensor of float32. shape=(B, T, M). `M` is the number of features.
    '''

    return tf.one_hot(x,num_feats)
    pass


class Layer:
    '''Parent network layer class. Every layer should implement these methods.

    Keep this class empty.
    '''
    def net_in(self):
        pass

    def net_act(self):
        pass

    def forward(self, x):
        pass

    def backward(self, optimizer, loss, tape):
        pass


class InputLayer(Layer):
    '''Input layer of the RNN. One-hot codes the tokens (chars) in the current mini-batch of inputs.'''
    def __init__(self, M):
        '''Input Layer constructor.
        
        Parameters:
        -----------
        M: int. Number of input neurons/features.
        '''
        self.M = M
        pass

    def forward(self, x):
        '''Performs forward pass through the input layer, which amounts to one-hot coding the current mini-batch.
        
        Parameters:
        -----------
        x: ndarray or tf Tensor. shape=(B, T). Int-coded chars/tokens in each sequence in the mini-batch.

        Returns:
        -----------
        tf Tensor of float32. shape=(B, T, M). One-hot coded chars/tokens in each sequence in the mini-batch.
        '''

        return one_hot(x,self.M)
        pass


class EmbeddingLayer(Layer):
    '''Creates a `H_e`-dimensional embedding of the input signal. Uses identity/linear activation function.'''
    def __init__(self, embedding_sz, num_neurons_prev_layer):
        '''Embedding Layer constructor

        Method should initialize the layer weights and bias.

        Parameters:
        -----------
        embedding_sz: int. Number of neurons in the current layer (H_e).
        num_neurons_prev_layer: int. Number of neurons in the layer below.

        NOTE: You should be using He/Kaiming initialization for the wts/bias. Check the notebook for a refresher on the
        equation.
        '''
       
        self.wts = tf.Variable(tf.random.normal((num_neurons_prev_layer, embedding_sz),0,(1/np.sqrt(num_neurons_prev_layer))))
        self.b = tf.Variable(tf.random.normal((embedding_sz,),0, 1/np.sqrt(num_neurons_prev_layer)))
        pass

    def get_wts(self):
        '''Returns the layer wts.
        
        Returns:
        -----------
        tf Tensor. shape=(M, H_e)
        '''
        return self.wts

    def get_bias(self):
        '''Returns the layer bias.
        
        Returns:
        -----------
        tf Tensor. shape=(H_e,)
        '''
        return self.b
        
    
    def set_wts(self, wts):
        '''Replaces the layer weights with `wts`.'''
        self.wts = wts
        
    
    def set_bias(self, bias):
        '''Replaces the layer bias with `bias`.'''
        self.b = bias

    def net_in(self, x):
        '''Computes the layer dense net input.
        
        Parameters:
        -----------
        x: tf Tensor. shape=(B*T, M). Input signal.

        Returns:
        -----------
        tf Tensor. shape=(B*T, H_e). The net input.
        '''
        
      
        net_in = x@self.wts + self.b
        return net_in

    def net_act(self, net_in):
        '''Computes the layer identity/linear net activation.
        
        Parameters:
        -----------
        net_in: tf Tensor. shape=(B*T, H_e). Net input.

        Returns:
        -----------
        tf Tensor. shape=(B*T, H_e). Net activation.
        '''
        return net_in

    def forward(self, x):
        '''Forward pass through the embedding layer.
        
        Parameters:
        -----------
        net_in: tf Tensor. shape=(B, T, M). Input to embedding layer.

        Returns:
        -----------
        tf Tensor. shape=(B, T, H_e). Net activation.

        NOTE: Pay close attention to shapes.
        '''
        b, t, m = x.shape[0],x.shape[1],x.shape[2]
        x = tf.reshape(x, (b*t, m))
        net_in = self.net_in(x)
        net_act = self.net_act(net_in)
        net_act = tf.reshape(net_act, (b, t, self.wts.shape[1]))
        return net_act 
        

    def backward(self, optimizer, loss, tape):
        '''Updates the wts/bias in the embedding layer through the backward pass.

        Parameters:
        -----------
        optimizer: tf Optimizer. TensorFlow optimizer object.
        loss: tf Tensor of scalar float. Average loss across current mini-batch at the end of the forward pass.
        tape: tf GradientTape. TensorFlow tape that has the wt/bias gradients recorded in it.
        '''
        wts = self.get_wts()
        b = self.get_bias()
        d_wts  = tape.gradient(loss, wts)
        d_b = tape.gradient(loss, b)
        optimizer.apply_gradients(zip([d_wts],[wts]))
        optimizer.apply_gradients(zip([d_b], [b]))
        pass


class GRULayer(Layer):
    '''Layer of Gated Recurrent Units (GRU)'''
    def __init__(self, num_neurons, num_neurons_prev_layer):
        '''GRULayer constructor
        
        Method should initialize the layer weights and bias.

        Parameters:
        -----------
        num_neurons: int. Number of neurons in the current layer (H_GRU).
        num_neurons_prev_layer: int. Number of neurons in the layer below.

        NOTE:
        - You should be using He/Kaiming initialization for the wts/bias. Check the notebook for a refresher on the
        equation.
        - There are quite a few weights/biases to initialize!! Use a helpful naming scheme!
        - For the test code to work, you should initialize each set of layer parameters in the following order:
            1. Update gate
            2. Reset gate
            3. GRU "y" related to candidate y netin/netact
        - For each of the above items, you should generate values in the following order:
            1. Feedforward
            2. Recurrent
            3. Bias
        '''
        self.up_gate_forward_wts = tf.Variable(tf.random.normal((num_neurons_prev_layer, num_neurons),0,(1/np.sqrt(num_neurons_prev_layer)))) 
        self.up_gate_recurr_wts = tf.Variable(tf.random.normal((num_neurons, num_neurons),0,(1/np.sqrt(num_neurons)))) 
        self.up_gate_b = tf.Variable(tf.random.normal((num_neurons,),0, 1/np.sqrt(num_neurons)))
        
        self.reset_gate_forward_wts = tf.Variable(tf.random.normal((num_neurons_prev_layer, num_neurons),0,(1/np.sqrt(num_neurons_prev_layer)))) 
        self.reset_gate_recurr_wts = tf.Variable(tf.random.normal((num_neurons, num_neurons),0,(1/np.sqrt(num_neurons)))) 
        self.reset_gate_b = tf.Variable(tf.random.normal((num_neurons,),0, 1/np.sqrt(num_neurons)))
        
        self.gru_y_forward_wts = tf.Variable(tf.random.normal((num_neurons_prev_layer, num_neurons),0,(1/np.sqrt(num_neurons_prev_layer)))) 
        self.gru_y_recurr_wts = tf.Variable(tf.random.normal((num_neurons, num_neurons),0,(1/np.sqrt(num_neurons)))) 
        self.gru_y_b = tf.Variable(tf.random.normal((num_neurons,),0, 1/np.sqrt(num_neurons)))
        # Placeholder for last state from the previous mini-batch.
        # Cannot be initialized here because GRULayer does not yet have access to the mini-batch size.
        self.last_state = None 

    def get_update_gate_wts_b(self):
        '''Returns the wts/bias related to the update gates.
        
        Returns:
        -----------
        tf Tensor. shape=(H_e, H_GRU). Input -> Recurrent update gate wts.
        tf Tensor. shape=(H_GRU, H_GRU). Recurrent -> Recurrent update gate wts.
        tf Tensor. shape=(H_GRU,). Update gate bias.
        '''
        return self.up_gate_forward_wts, self.up_gate_recurr_wts, self.up_gate_b
        

    def get_reset_gate_wts_b(self):
        '''Returns the wts/bias related to the reset gates.
        
        Returns:
        -----------
        tf Tensor. shape=(H_e, H_GRU). Input -> Recurrent reset gate wts.
        tf Tensor. shape=(H_GRU, H_GRU). Recurrent -> Recurrent reset gate wts.
        tf Tensor. shape=(H_GRU,). Reset gate bias.
        '''
        return self.reset_gate_forward_wts, self.reset_gate_recurr_wts, self.reset_gate_b
        

    def get_candidate_wts_b(self):
        '''Returns the wts/bias related to the GRY "y" candidate activation.
        
        Returns:
        -----------
        tf Tensor. shape=(H_e, H_GRU). Input -> Recurrent candidate activation wts.
        tf Tensor. shape=(H_GRU, H_GRU). Recurrent -> Recurrent candidate activation wts.
        tf Tensor. shape=(H_GRU,). Candidate activation bias.
        '''
        return self.gru_y_forward_wts, self.gru_y_recurr_wts, self.gru_y_b
        
    
    def set_update_gate_wts_b(self, u_wts_i2h, u_wts_h2h, u_b):
        '''Replaces the update gate parameters with those passed in.

        Parameters:
        -----------
        u_wts_i2h: tf Tensor. shape=(H_e, H_GRU). New input -> Recurrent update gate wts.
        u_wts_h2h: tf Tensor. shape=(H_GRU, H_GRU). New recurrent -> Recurrent update gate wts.
        u_b: tf Tensor. shape=(H_GRU,). New update gate bias.
        '''
        self.up_gate_forward_wts = u_wts_i2h
        self.up_gate_recurr_wts = u_wts_h2h
        self.up_gate_b = u_b
        
    
    def set_reset_gate_wts_b(self, r_wts_i2h, r_wts_h2h, r_b):
        '''Replaces the reset gate parameters with those passed in.

        Parameters:
        -----------
        r_wts_i2h: tf Tensor. shape=(H_e, H_GRU). New input -> Recurrent reset gate wts.
        r_wts_h2h: tf Tensor. shape=(H_GRU, H_GRU). New recurrent -> Recurrent reset gate wts.
        r_b: tf Tensor. shape=(H_GRU,). New reset gate bias.
        '''
        self.reset_gate_forward_wts = r_wts_i2h
        self.reset_gate_recurr_wts = r_wts_h2h
        self.reset_gate_b = r_b
    
    def set_candidate_wts_b(self, c_wts_i2h, c_wts_h2h, c_b):
        '''Replaces the candidate activation parameters with those passed in.

        Parameters:
        -----------
        c_wts_i2h: tf Tensor. shape=(H_e, H_GRU). New input -> Recurrent candidate activation wts.
        c_wts_h2h: tf Tensor. shape=(H_GRU, H_GRU). New recurrent -> Recurrent candidate activation wts.
        c_b: tf Tensor. shape=(H_GRU,). New candidate activation bias.
        '''
        self.gru_y_forward_wts =c_wts_i2h
        self.gru_y_recurr_wts = c_wts_h2h
        self.gru_y_b = c_b

    def get_initial_state(self, B):
        '''Gets the initialization state of the GRULayer, which is all 0 activation.

        Parameters:
        -----------
        B: int. Number of sequences in the mini-batch.

        Returns:
        -----------
        tf Tensor. shape=(B, H_GRU). Initial state/activations of the GRU layer.
        '''
        self.last_state = tf.zeros((B,self.reset_gate_b.shape[0]))
        return self.last_state
        

    def reset_state(self, B):
        '''Resets/reinitalizes the state of the GRULayer to all 0 activations.

        Parameters:
        -----------
        B: int. Number of sequences in the mini-batch.
        '''
        self.last_state = tf.zeros((B,self.reset_gate_b.shape[0]))
        

    def get_last_state(self, B):
        '''Returns the last state/activations of the GRULayer.

        Parameters:
        -----------
        B: int. Number of sequences in the mini-batch.

        Returns:
        -----------
        tf Tensor. shape=(B, H_GRU). Last state/activations of the GRU layer.

        NOTE: If the last state has not been initialized yet, the initial state should be returned as the last state.
        '''
        if self.last_state == None:
            return self.get_initial_state(B)
        else:
            return self.last_state 
        

    def set_last_state(self, new_last_state):
        '''Replaces the last state with a new state `new_last_state`.

        Parameters:
        -----------
        tf Tensor. shape=(B, H_GRU). New state/activations that should become the last state of the GRU layer.
        '''
        self.last_state = new_last_state
        

    def net_in(self, x, prev_net_act):
        '''Computes the net input of the GRU Layer for one current time in the current mini-batch.

        Parameters:
        -----------
        x: tf Tensor. shape=(B, H_e). Current time step of the current mini-batch signal from the previous layer below.
        prev_net_act: tf Tensor. shape=(B, H_GRU). The GRU net_act from the previous time step.

        Returns:
        -----------
        tf Tensor. shape=(B, H_GRU). Net input for the update gate of all units in the GRU layer.
        tf Tensor. shape=(B, H_GRU). Net input for the reset gate of all units in the GRU layer.
        tf Tensor. shape=(B, H_GRU). Net input for the candidate act of all units in the GRU layer.

        NOTE:
        - Don't forget that there are both feedforward AND recurrent connections in this layer.
        - Don't forget to defer a component of the GRU y candidate netin
        '''
        
        u_net_in = x @ self.up_gate_forward_wts + prev_net_act@self.up_gate_recurr_wts + self.up_gate_b
        r_net_in = x @ self.reset_gate_forward_wts + prev_net_act@self.reset_gate_recurr_wts + self.reset_gate_b
        c_net_in = x @ self.gru_y_forward_wts + self.gru_y_b
        
        return u_net_in, r_net_in, c_net_in

    def net_act(self, update_gate_in, reset_gate_in, cand_in, prev_net_act):
        '''Computes the net activation of the GRU Layer for one current time in the current mini-batch.

        Parameters:
        -----------
        update_gate_in: tf Tensor. shape=(B, H_GRU). Net input for the update gate of all units in the GRU layer.
        reset_gate_in: tf Tensor. shape=(B, H_GRU). Net input for the reset gate of all units in the GRU layer.
        cand_in: tf Tensor. shape=(B, H_GRU). Net input for the candidate act of all units in the GRU layer.
        prev_net_act: tf Tensor. shape=(B, H_GRU). net_act from the previous time step.
        
        Returns:
        -----------
        tf Tensor. shape=(B, H_GRU). The GRU net_act computed for the current time step.
        tf Tensor. shape=(B, H_GRU). The update gate net_act computed for the current time step.
        tf Tensor. shape=(B, H_GRU). The reset gate computed for the current time step.
        '''
        u_act = tf.sigmoid(update_gate_in)
        r_act = tf.sigmoid(reset_gate_in)
        act_c = cand_in + (r_act * prev_net_act) @ self.gru_y_recurr_wts
        y_c_act = tf.tanh(act_c)
        y_netAct = (1-u_act)*prev_net_act + u_act * y_c_act
        
        return y_netAct, u_act, r_act


    def forward(self, x):
        '''Forward pass through the GRU layer.

        Parameters:
        -----------
        x: tf Tensor. shape=(B, T, H_e). Input signal coming from the Embedding layer below.

        Returns:
        -----------
        tf Tensor. shape=(B, T, H_GRU). GRU net_act at all time steps in current mini-batch.

        TODO:
        1. Starting with the GRU net_act from the last time step from the previous mini-batch (i.e. last state), compute
        the net input and net activations sequentially across time. 
        2. Before the forward pass ends, don't forget to update the last state value to the net_act at the final time step
        in the current mini-batch.

        HINT:
        To get around issues with TF assignment, tf.stack might be helpful...
        '''
        b, t, h_e, h_gru = x.shape[0], x.shape[1], x.shape[2], self.gru_y_recurr_wts.shape[0]
        net_acts = []
        for i in range(t):
            y = x[:,i, :]
           
            u_net_in, r_net_in, c_net_in = self.net_in(y, self.get_last_state(b))
            y_netAct, u_act, r_act = self.net_act(u_net_in, r_net_in, c_net_in, self.get_last_state(b))
            
            net_acts.append(y_netAct)
        
            self.set_last_state(y_netAct)
      
        return tf.stack(net_acts, axis=1)

    def backward(self, optimizer, loss, tape):
        '''Updates the wts/bias in the GRU layer through the backward pass.

        Parameters:
        -----------
        optimizer: tf Optimizer. TensorFlow optimizer object.
        loss: tf Tensor of scalar float. Average loss across current mini-batch at the end of the forward pass.
        tape: tf GradientTape. TensorFlow tape that has the wt/bias gradients recorded in it.
        '''
        # g_f, g_r, g_b = self.get_update_gate_wts_b()
        # r_f, r_r, r_b = self.get_reset_gate_wts_b()
        # c_f, c_r, c_b = self.get_candidate_wts_b()
        
        # d_g_f  = tape.gradient(loss, g_f)
        # d_g_r  = tape.gradient(loss, g_r)
        # d_g_b  = tape.gradient(loss, g_b)
        
        # d_r_f  = tape.gradient(loss, r_f)
        # d_r_r  = tape.gradient(loss, r_r)
        # d_r_b  = tape.gradient(loss, r_b)
        
        # d_c_f  = tape.gradient(loss, c_f)
        # d_c_r  = tape.gradient(loss, c_r)
        # d_c_b  = tape.gradient(loss, c_b)
        
        # optimizer.apply_gradients(zip([d_g_f],[g_f]))
        # optimizer.apply_gradients(zip([d_g_r],[g_r]))
        # optimizer.apply_gradients(zip([d_g_b],[g_b]))
        
        # optimizer.apply_gradients(zip([d_r_f],[r_f]))
        # optimizer.apply_gradients(zip([d_r_r],[r_r]))
        # optimizer.apply_gradients(zip([d_r_b],[r_b]))
        
        # optimizer.apply_gradients(zip([d_c_f],[c_f]))
        # optimizer.apply_gradients(zip([d_c_r],[c_r]))
        # optimizer.apply_gradients(zip([d_c_b],[c_b]))

        u_wts_i2h, u_wts_h2h, u_b = self.get_update_gate_wts_b()
        r_wts_i2h, r_wts_h2h, r_b = self.get_reset_gate_wts_b()
        c_wts_i2h, c_wts_h2h, c_b = self.get_candidate_wts_b()
    
        d_wts_all = tape.gradient(loss, [u_wts_i2h, u_wts_h2h, u_b, r_wts_i2h, r_wts_h2h, r_b, c_wts_i2h, c_wts_h2h, c_b ])
        optimizer.apply_gradients(zip(d_wts_all, [u_wts_i2h, u_wts_h2h, u_b, r_wts_i2h, r_wts_h2h, r_b, c_wts_i2h, c_wts_h2h, c_b ]))
      
      


class DenseLayer(Layer):
    '''Dense layer that uses softmax activation function (we are assuming this is the output layer).'''
    def __init__(self, num_neurons, num_neurons_prev_layer):
        '''Dense Layer constructor

        Method should initialize the layer weights and bias.

        Parameters:
        -----------
        num_neurons: int. Number of neurons in the current layer.
        num_neurons_prev_layer: int. Number of neurons in the layer below (H_GRU).

        NOTE: You should be using He/Kaiming initialization for the wts/bias. Check the notebook for a refresher on the
        equation.
        '''
        self.wts = tf.Variable(tf.random.normal((num_neurons_prev_layer, num_neurons),0,(1/np.sqrt(num_neurons_prev_layer)))) 
        self.b = tf.Variable(tf.random.normal((num_neurons,),0, 1/np.sqrt(num_neurons)))

    def get_wts(self):
        '''Returns the layer wts.
        
        Returns:
        -----------
        tf Tensor. shape=(H_GRU, C)
        '''
        return self.wts

    def get_bias(self):
        '''Returns the layer bias.
        
        Returns:
        -----------
        tf Tensor. shape=(C,)
        '''
        return self.b
    
    def set_wts(self, wts):
        '''Replaces the layer weights with `wts`.'''
        
        self.wts = wts
    
    def set_bias(self, bias):
        '''Replaces the layer bias with `bias`.'''
        
        self.b = bias

    def net_in(self, x):
        '''Computes the layer dense net input.
        
        Parameters:
        -----------
        x: tf Tensor. shape=(B*T, H_GRU). Input signal.

        Returns:
        -----------
        tf Tensor. shape=(B*T, C). The net input.
        '''
        return x@self.wts+self.b

    def net_act(self, net_in):
        '''Computes the softmax activation.
        
        Parameters:
        -----------
        net_in: tf Tensor. shape=(B*T, C). Net input.

        Returns:
        -----------
        tf Tensor. shape=(B*T, C). Net activation.
        '''
        # z_max = np.max(net_in, axis = 1, keepdims=True)
        # adj_net = net_in - z_max

        # net_act = np.exp(adj_net)/np.sum(np.exp(adj_net), axis=1, keepdims=True)
        # return net_act

        return tf.nn.softmax(net_in)
        

    def forward(self, x):
        '''Forward pass through the Dense layer.
        
        Parameters:
        -----------
        net_in: tf Tensor. shape=(B, T, H_GRU). Input to Dense layer.

        Returns:
        -----------
        tf Tensor. shape=(B, T, C). NET INPUT for the current mini-batch signal `x`.

        NOTE: Pay close attention to shapes.
        '''
        b, t, h_gru = x.shape[0], x.shape[1], x.shape[2] 
        c = self.get_bias().shape[0]
       

        x = tf.reshape(x, (b*t, h_gru))
       
        net_in = self.net_in(x)
       
        act = self.net_act(net_in)
        
        net_in= tf.reshape(net_in, (b,t,c))
        act = tf.reshape(act, (b,t,c))
        
        return net_in
        

    def backward(self, optimizer, loss, tape):
        '''Updates the wts/bias in the Dense layer through the backward pass.

        Parameters:
        -----------
        optimizer: tf Optimizer. TensorFlow optimizer object.
        loss: tf Tensor of scalar float. Average loss across current mini-batch at the end of the forward pass.
        tape: tf GradientTape. TensorFlow tape that has the wt/bias gradients recorded in it.
        '''
        wts = self.get_wts()
        b = self.get_bias()
        d_wts  = tape.gradient(loss, wts)
        d_b = tape.gradient(loss, b)
        optimizer.apply_gradients(zip([d_wts],[wts]))
        optimizer.apply_gradients(zip([d_b], [b]))
        


class RNN:
    '''Recurrent neural network with the following architecture:
    
    Input Layer → Embedding Layer (identity/linear act) → GRU Layer → Dense output layer (softmax act)
    
    '''
    def __init__(self, vocab_sz, embedding_sz, num_gru_neurons, load_wts=False):
        '''RNN constructor

        This method should build the layers in the network.

        Parameters:
        -----------
        vocab_sz: int. Number of tokens in the vocabulary.        
        embedding_sz: int. The embedding size/dimension in the embedding layer.
        num_gru_neurons: int. Number of neurons in the GRU layer.
        load_wts: bool. Do we load the net wts/biases from file?
        '''
        self.vocab_sz = vocab_sz
        #self.embedding_sz = embedding_sz
        #self.num_gru_neurons = num_gru_neurons
        self.In_layer = InputLayer(vocab_sz)
        self.embed_layer = EmbeddingLayer(embedding_sz, vocab_sz)
        self.GRU_layer = GRULayer(num_gru_neurons, embedding_sz)
        self.dense_layer = DenseLayer(vocab_sz, num_gru_neurons)
        # Keep me
        if load_wts:
            self.load_wts()

    def get_embedding_layer(self):
        '''Returns the Embedding Layer object in the network'''
        
        return self.embed_layer

    def get_gru_layer(self):
        '''Returns the GRU Layer object in the network'''
        
        return self.GRU_layer
    
    def get_output_layer(self):
        '''Returns the output Dense layer object in the network'''
        
        return self.dense_layer

    def forward(self, x):
        '''Forward pass through the network with mini-batch `x`.
        
        Parameters:
        -----------
        x: ndarray or tf Tensor. shape=(B, T).

        Returns:
        -----------
        tf Tensor. shape=(B, T, C). net_in of the output Dense layer.
        '''
        out_1 = self.In_layer.forward(x)
        out_2 = self.embed_layer.forward(out_1)
        out_3 = self.GRU_layer.forward(out_2)
        out_4 = self.dense_layer.forward(out_3)
        
        return out_4

    def loss(self, out_net_in, y_batch):
        '''Computes the cross-entropy loss averaged over the current mini-batch.
       
        Parameters:
        -----------
        out_net_in: tf Tensor. shape=(B, T, C). net_in of the output Dense layer.
        y_batch: ndarray or tf Tensor. shape=(B, T). int-coded class labels for all time steps in the current mini-batch.
        
        Returns:
        -----------
        tf Tensor of scalar float. The cross-entropy loss averaged over the current mini-batch.
        '''
        B,T,C = out_net_in.shape
        out_net_in_flat = tf.reshape(out_net_in, [B*T,C])
        y_batch_flat = tf.reshape(y_batch, [B*T,])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch_flat, out_net_in_flat, from_logits=True)

       
        loss = tf.reduce_mean(loss)
       
        
        #return loss
        return loss
    

    def fit(self, x, y, epochs=1, lr=1e-3, verbose=True, print_every_epoch=1, print_every_batch=50):
        '''Trains the RNN on the int-coded mini-batches `x` and corresponding int-coded class labels `y`.
        Uses the Adam optimizer.

        Parameters:
        -----------
        x: ndarray or tf Tensor. shape=(num_seqs, B, T). Int-coded sequences organized in mini-batches.
        y: ndarray or tf Tensor. shape=(num_seqs, B, T). Int-coded sequence labels organized in mini-batches.
        epochs: int. Number of epochs over which the RNN is trained.
        lr: float. Learning rate of the optimizer.
        verbose: bool. If True, loss progress printouts appear every `print_every_epoch` epochs and within each epoch
            every `print_every_batch` mini-batches. If False, no loss printouts should appear.
        print_every_epoch. int. How often in epochs should the average loss across mini-batches within an epoch be printed
            out?
        print_every_batch. int. How often during every epoch should the loss for the current mini-batch be printed out?
        
        Returns:
        -----------
        Python list. len=epochs. The loss averaged across all mini-batches within each epoch.

        NOTE:
        - This is a fairly standard training loop. There is no checking for validation loss, however.
        - Use the provided code to call `save_wts` just as every epoch finishes to save off the current network wts to file
        for backup purposes.
        - Because each layer is updating its own wts/biases from the TF gradient tape, we need the gradients in the tape
        to persistent across multiple calls of the `gradient` method (recall, by default the tape deletes any gradients
        after a single call). To allow the gradients to persistent, when creating the gradient tape object, pass in the
        keyword argument: persistent=True .
        '''
        num_seqs, B,T = x.shape
        self.GRU_layer.reset_state(B)
        optim = tf.keras.optimizers.legacy.Adam(learning_rate = lr)
        total_loss_hist = []
        for epoch in range(epochs): #for epoch epochs
            epoch_loss_hist = []
            for batch in range(num_seqs):#for each batch
                batch_x = x[batch]
                batch_y = y[batch]
                with tf.GradientTape(persistent = True) as tape:
                    net_act = self.forward(batch_x)#handles the time element
                    loss = self.loss(net_act,batch_y)
                epoch_loss_hist.append(loss)
                if batch%print_every_batch == 0 and batch!=0:
                    print("batch: ",batch," loss: ",float(loss))

                self.backward(optim,loss,tape)
            average_loss = sum(epoch_loss_hist)/len(epoch_loss_hist)
            if epoch % print_every_epoch == 0: 
                print("epoch: ",epoch," average loss: ",float(average_loss))
            total_loss_hist.append(average_loss)
        return total_loss_hist
                


    
    def backward(self, optimizer, loss, tape):
        '''Backward pass through network to update wts/biases in each layer.
        
        Parameters:
        -----------
        optimizer: tf Optimizer. TensorFlow optimizer object.
        loss: tf Tensor of scalar float. Average loss across current mini-batch at the end of the forward pass.
        tape: tf GradientTape. TensorFlow tape that has the wt/bias gradients recorded in it.

        TODO: Traverse the layers top → bottom (from output back to input layer), calling respective backward layer methods.
        '''
        self.dense_layer.backward(optimizer, loss, tape)
        self.GRU_layer.backward(optimizer, loss, tape)
        self.embed_layer.backward(optimizer, loss, tape)
        self.In_layer.backward(optimizer, loss, tape)
        pass

    def save_wts(self, filename='wts.npz'):
        '''Saves all wts/biases to the file in the project folder with the provided filename.
        
        This is provided to you / should not require modification.
        '''
        # Get embedding layer wts/b
        y_wts = self.get_embedding_layer().get_wts()
        y_b = self.get_embedding_layer().get_bias()

        # Get GRU layer wts/b
        u_wts_i2h, u_wts_h2h, u_b = self.get_gru_layer().get_update_gate_wts_b()
        r_wts_i2h, r_wts_h2h, r_b = self.get_gru_layer().get_reset_gate_wts_b()
        c_wts_i2h, c_wts_h2h, c_b = self.get_gru_layer().get_candidate_wts_b()

        # Get Dense layer wts/b
        z_wts = self.get_output_layer().get_wts()
        z_b = self.get_output_layer().get_bias()

        np.savez_compressed(filename,
                            y_wts=y_wts,
                            y_b=y_b,
                            u_wts_i2h=u_wts_i2h,
                            u_wts_h2h=u_wts_h2h,
                            u_b=u_b,
                            r_wts_i2h=r_wts_i2h,
                            r_wts_h2h=r_wts_h2h,
                            r_b=r_b,
                            c_wts_i2h=c_wts_i2h,
                            c_wts_h2h=c_wts_h2h,
                            c_b=c_b,
                            z_wts=z_wts,
                            z_b=z_b)
        
    def load_wts(self, filename='wts.npz'):
        '''Loads all wts/biases from the file in the project folder with the provided filename.
        
        This is provided to you / should not require modification.
        '''
        wts_dict = np.load(filename)
        
        # Restore Embedding layer wts/bias
        self.get_embedding_layer().set_wts(tf.Variable(wts_dict['y_wts']))
        self.get_embedding_layer().set_bias(tf.Variable(wts_dict['y_b']))
        
        # Restore GRU wts/bias
        u_wts_i2h, u_wts_h2h, u_b = wts_dict['u_wts_i2h'], wts_dict['u_wts_h2h'], wts_dict['u_b']
        self.get_gru_layer().set_update_gate_wts_b(tf.Variable(u_wts_i2h), tf.Variable(u_wts_h2h), tf.Variable(u_b))
        r_wts_i2h, r_wts_h2h, r_b = wts_dict['r_wts_i2h'], wts_dict['r_wts_h2h'], wts_dict['r_b']
        self.get_gru_layer().set_reset_gate_wts_b(tf.Variable(r_wts_i2h), tf.Variable(r_wts_h2h), tf.Variable(r_b))
        c_wts_i2h, c_wts_h2h, c_b = wts_dict['c_wts_i2h'], wts_dict['c_wts_h2h'], wts_dict['c_b']
        self.get_gru_layer().set_candidate_wts_b(tf.Variable(c_wts_i2h), tf.Variable(c_wts_h2h), tf.Variable(c_b))

        # Restore Dense layer wts/bias
        self.get_output_layer().set_wts(tf.Variable(wts_dict['z_wts']))
        self.get_output_layer().set_bias(tf.Variable(wts_dict['z_b']))

    def generate_sequence(self, prompt, length, char2ind_map, ind2char_map, plot_probs=False):
        '''Generates/predicts a sequence of chars of length `length` chars that follow the provided prompt.
        It is helpful remember that the RNN generates chars one at a time sequentially. Therefore in 
        prediction/generation mode, the network processes tokens in mini-batches of one item for one time step.

        Parameters:
        -----------
        prompt: str. Chars to pass thru the RNN one-at-a-time sequentially before the net predicts the next char.
        length: int. Number of chars that RNN generates after the prompt chars. 
        char2ind_map: Python dictionary. Keys: chars in vocab. Values: int code of a char in the vocab.
        ind2char_map: Python dictionary. Keys: int code of a char in the vocab. Values: Which char it corresponds to in
            the vocab.

        Returns:
        -----------
        str. len=(len(prompt) + length). The provided prompt concatenated with the set of RNN generated chars.

        TODO:
        1. Before generating anything with the RNN, first reset the GRU state to prevent whatever was processed last from
        influencing what is generated.
        2. Have the network process all int-coded tokens in the prompt sequentially, EXCEPT for the last one.
        The purpose of this is to establish the GRU's state in the context of the prompt. Be careful about shapes!
        3. Have the network operate in a feedback loop: the char predicted from the previous time step becomes the input
        to the net on the next time step. The starting point for this 2nd phase is the last char from the prompt, that
        before this point, should not have yet been processed by the network. Be careful about shapes! To get the next
        predicted char:
            - Compute the output layer netAct (softmax) for the current char. This will return the softmax probability of
            which char should come next in the generated sequence.
            - Squeeze this (C,) prob distribution and convert back to numpy.
            - Use `np.random.choice` with the `p` keyword argument to pick the index of the most likely char that comes next
            in proportion of the softmax probability of each char.
        4. Convert the int-coded tokens generated by the RNN (`length` ints in total) back to chars, then concatenate
        with the prompt before returning the resultant string.
        '''
        
        #reset GRU state: 
        self.GRU_layer.reset_state(1)
        
        promt_1 = prompt[:-1]
        
        # loop over entire prompt except last char and int code it and pass it into forward with shape (1,1)
        for c in promt_1:
            c_ind = char2ind_map[c]
            c_arr = np.array([[c_ind]])
       
            self.forward(c_arr)
        
        # set full prompt to the orginal prompt
        full_prompt = prompt
        
        # add length chars to full prompt
        for gen in range(length):
            
            # get last char
            c_last = full_prompt[-1]
           
            
            #int code it
            c_last_ind = char2ind_map[c_last]
            
            #make into array (1,1)
            c_last_arr = np.array([[c_last_ind]])
            
            # pass through forward then through softmax 
            net_in_Dense = self.forward(c_last_arr)
          
            net_act = self.dense_layer.net_act(net_in_Dense)
           
            
            # squeeze results
            prob = tf.squeeze(net_act).numpy()
           
            
            # get char ind
            c_next_ind = np.random.choice(len(char2ind_map), p=prob)
            # get char from mapping
            c_next = ind2char_map[c_next_ind]
            #add it to full prompt
            full_prompt += c_next
            
        return full_prompt
            
            
        

        
        pass
