'''snn.py
Layers of spiking LIF neurons organized in a neural network that performs classification
YOUR NAMES HERE
CS 443: Bio-Inspired Machine Learning
Project 4: Spiking Neural Networks
'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from viz import draw_grid_image, plot_voltage

class Layer:
    '''Generic Layer, Parent class to specific layer types.'''
    def __init__(self, num_neurons, params):
        '''Constructor
        
        Parameters:
        -----------
        num_neurons: int. Number of neurons in the current layer.
        params: Python dictionary. Holds parameters from the JSON config file appropriate for the current layer.
            NOTE: The "subdictionary" for the current layer should be passed in so that parameters are accessed like:
                params['a_param']
            instead of:
                params['curr_layer']['a_param']

        TODO: Create instance variables for number of neurons and the param dictionary.
        '''
        self.num_neurons = num_neurons
        self.params = params
        pass
        
     

    def initialize_state(self):
        '''Initializes the state of the neurons, which means
        - Allocating the spikes of each neuron to all False/0.
        - Allocating the trace of each neuron to the 0.
        '''
        self.spikes = np.zeros((self.num_neurons,)).astype(bool)
        self.trace = np.zeros((self.num_neurons,))
        pass

         
        
        
    def reset_state(self):
        '''Resets the state of the LIF neurons, which means:
        - Resetting the spikes of each neuron to all False/0.
        - Resetting the trace of each neuron to 0.
        '''
        #may want to be inplace?
        self.spikes = np.zeros((self.num_neurons,)).astype(bool)
        self.trace = np.zeros((self.num_neurons,))
        pass
        
        
      

    def get_num_units(self):
        '''Gets the number of neurons in the layer'''
        
        return self.num_neurons

    def get_spikes(self):
        '''Gets the spikes at the current time step for neurons in the layer
        
        Returns:
        -----------
        ndarray of bool. shape=(num_neurons,).
        '''
        return self.spikes
      

    def get_trace(self):
        '''Gets the trace at the current time step for neurons in the layer
        
        Returns:
        -----------
        ndarray. shape=(num_neurons,).
        '''
        return self.trace
    

    def set_spikes(self, spikes):
        '''Replaces the spike state at the current time step for all neurons in the layer with `spikes`
        
        Used for debugging/test code
        
        Parameters:
        -----------
        spikes: ndarray of bool. shape=(num_neurons,).
        '''
        self.spikes = spikes

    def set_trace(self, trace):
        '''Replaces the trace state at the current time step for all neurons in the layer with `trace`
        
        Used for debugging/test code
        
        Parameters:
        -----------
        trace: ndarray. shape=(num_neurons,).
        '''
        self.trace = trace

class InputLayer(Layer):
    '''Input layer to the spiking neural network. Takes data samples in (N=num_samps, M=num_feats) format and encodes
    it into Poisson spike train (temporal rate coding). Neurons in the layer have the properties:
    
    1. Trace'''
    def __init__(self, params):
        '''Constructor
        
        Parameters:
        -----------
        num_neurons: int. Number of neurons in the input layer.
        params: Python dictionary. Holds parameters from the JSON config file appropriate for the current layer.
            NOTE: The "subdictionary" for the current layer should be passed in so that parameters are accessed like:
                params['a_param']
            instead of:
                params['curr_layer']['a_param']

        TODO:
        1. Call the superclass constructor
        2. Initialize the state.
        '''
        super().__init__(params["num_neurons"], params)
        self.initialize_state()
        


    def get_num_time_steps(self):
        return self.params["num_time_steps"]  # TODO: Adapt to your naming conventions

    def poisson_encode(self, x):
        '''Encode the input data `x` as a Poisson spike train over `t_max` time steps, where each time step corresponds
        to 1 msec of real time.

        Parameters:
        -----------
        x: ndarray. (N, M). Dataset with `N` samples and `M` features.

        Returns:
        -----------
        ndarray of bool. (N, T, M). Dataset encoded with N samples, T time steps, and M features.

        TODO: Copy-paste the code that you wrote during lab. Adapt it so that it accesses params from the dictionary.
        '''
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        N, M = x.shape
        
        x_a = np.reshape(x/1000, (N,1,M))


        randoms = np.random.uniform(0,1,(N,self.params["num_time_steps"],M)) 

        return (0 >= (randoms - x_a))


    def forward(self, x):
        '''Do the foward pass through the input layer.
         
        Parameters:
        -----------
        x: ndarray of bool. shape=(num_neurons,). The Poisson encoded current data sample at the current time step. 
            We are assuming that we will process the dataset SGD-style (one sample at a time). So we assume `x` coming
            with has N=1. To make life easier with indexing, the batch dimension is squeezed out.
            
        Returns:
        -----------
        ndarray of bool. (num_neurons,). Spikes at the current time step.
        
        TODO:
        1. Set the instance variable for the current set of spikes to the Poisson encoded input.
        2. Apply the trace based on the current spikes (remember this are 2 cases to consider — spike or no spike).
        '''
        self.spikes = x
        self.trace[self.spikes] = 1
        self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
        
        return self.spikes
        
        


class RecurrentLayer(Layer):
    '''Layer of LIF neurons with inhibitory recurrent interactions among the neurons in the layer.'''
    def __init__(self, num_neurons_prev_layer, params):
        '''Constructor
        
        Parameters:
        -----------
        num_neurons: int. Number of neurons in the current recurrent layer.
        num_neurons_prev_layer: int. Number of neurons in the layer below (e.g. Input layer).
        params: Python dictionary. Holds parameters from the JSON config file appropriate for the current layer.
            NOTE: The "subdictionary" for the current layer should be passed in so that parameters are accessed like:
                params['a_param']
            instead of:
                params['curr_layer']['a_param']

        TODO:
        1. Call the superclass constructor
        2. Initialize the Input Layer -> Recurrent Layer wts.
            shape=(num_neurons_prev_layer, num_neurons).
            Random numbers between 0 and w_max, where w_max is a parameter defined in the JSON config file.
        3. Initialize the Recurrent Layer -> Recurrent Layer wts (recurrent interaction wts).
            shape=(num_neurons, num_neurons).
            Each neuron sends itself no recurrent signal, but sends all other neurons in the layer an inhibitory signal
            with strength -g_inh, where g_inh is the gain on the inhibitory recurrent signals defined in the JSON config
            file.

            Example recurrent wts:
            [[0, -g_inh, -g_inh, ..., -g_inh],
             [-g_inh, 0, -g_inh, ..., -g_inh],
             [-g_inh, -g_inh, 0, ..., -g_inh],
             ...
             [-g_inh, -g_inh, -g_inh, ..., 0]]
        4. Initialize the state of all neurons in the layer.
        '''
        super().__init__(params["num_neurons"], params)
        self.in_rec_wts = np.random.uniform(0, params["max_wts"], size=(num_neurons_prev_layer,params["num_neurons"]))
        self.rec_rec_wts = np.ones((params["num_neurons"], params["num_neurons"]))*self.params["gain_wts"]*-1
        np.fill_diagonal(self.rec_rec_wts, 0)
        self.initialize_state()
        
        # Placeholder for boolean to indicate whether the layer network is training or in prediction mode
        self.is_learning = None  # KEEP ME
        

    def initialize_state(self):
        '''Initializes the state of the recurrent layer LIF neurons, which means
        - Allocating the spikes of each neuron to all False/0.
        - Allocating the trace of each neuron to the 0.
        - Allocating the voltage of each neuron to the resting voltage.
        - Allocating the refractory count of each neuron to 0.
        - Allocating the adaptive threshold of each neuron to all 0.
        '''
        # initialize trace + spikes
        super().initialize_state()
        
        self.v = self.params["v_rest"]*np.ones(self.params["num_neurons"])
        
        self.refrac_count = np.zeros(self.params["num_neurons"])
      
        self.adaptive_thres = np.zeros(self.params["num_neurons"])
        

    def reset_state(self):
        '''Resets the state of the recurrent layer LIF neurons, which means:
        - Resetting the spikes of each neuron to all False/0.
        - Resetting the trace of each neuron to 0.
        - Resetting the voltage of each neuron to its **resting voltage**.
        - Resetting the refractory count of each neuron to 0.

        NOTE: The adaptive thresholds do not reset. They persist long-term across samples.
        '''
        # spikes and trace
        super().reset_state()
        
        self.v = self.params["v_rest"]*np.ones(self.params["num_neurons"])
        
        self.refrac_count = np.zeros(self.params["num_neurons"])
      
     

    def reset_adaptive_threshold(self):
        '''Resets the adaptive threshold of each neuron to all 0.
        
        NOTE: This is rarely needed, perhaps when training on a brand new dataset.
        '''
        self.adaptive_thres = np.zeros(self.params["num_neurons"])
        

    def get_adaptive_threshold(self):
        '''Gets the adaptive threshold of each neuron.

        Returns:
        -----------
        ndarray. shape=(num_neurons). The current adaptive threshold of each neuron.
        '''
        return self.adaptive_thres

    def get_v(self):
        '''Gets the current voltage of each neuron.

        Returns:
        -----------
        ndarray. shape=(num_neurons). The current voltage of each neuron.
        '''
        return self.v

    def get_refractory_count(self):
        '''Gets the refractory count of each neuron — i.e. how much longer each neuron is in its refractory period.

        Returns:
        -----------
        ndarray. shape=(num_neurons). The refractory count of each neuron.
        '''
        return self.refrac_count

    def get_in_wts(self):
        '''Gets the Input Layer -> Recurrent layer wts.

        Returns:
        -----------
        ndarray. shape=(num_neurons_prev_layer, num_neurons). The Input Layer -> Recurrent layer wts.
        '''
        return self.in_rec_wts

    def get_recurrent_wts(self):
        '''Gets the Recurrent Layer -> Recurrent layer wts.

        Returns:
        -----------
        ndarray. shape=(num_neurons, num_neurons). The Recurrent Layer -> Recurrent layer wts.
        '''
        return self.rec_rec_wts
    
    def set_is_learning(self, mode):
        '''Sets boolean instance variable to indicate whether the layer is currently in training mode or prediction mode

        Parameters:
        -----------
        mode: bool. True if net in training mode, False if net in prediction mode.

        NOTE: See instance variable in constructor.
        '''
        self.is_learning = mode

    def set_v(self, v_new):
        '''Sets the voltages of the neurons to `v_new`

        Parameters:
        -----------
        v_new: ndarray. shape=(num_neurons). New voltage for each neuron in the layer.
        '''
        self.v = v_new

    def set_refractory_count(self, new_counts):
        '''Sets the refractory count of the neurons to `new_counts`

        Parameters:
        -----------
        new_counts: ndarray. shape=(num_neurons). New refractory count for each neuron in the layer.
        '''
        self.refrac_count = new_counts

    def set_in_wts(self, new_in_wts):
        '''Sets the Input Layer -> Recurrent Layer wts to `new_in_wts`

        Parameters:
        -----------
        new_in_wts: ndarray. shape=(num_neurons_prev_layer, num_neurons). New Input Layer -> Recurrent layer wts.
        '''
        self.in_rec_wts = new_in_wts

    def forward(self, x_in):
        '''Does the forward pass through the current Recurrent Layer, which means computing the net input and activations.

        Parameters:
        -----------
        x_in: ndarray. shape=(num_neurons_prev_layer,). Spikes produced by the layer below (e.g. the Input Layer).

        Returns:
        -----------
        ndarray of bool. shape=(num_neurons,). Spikes produced all neurons in the current layer at the current time step.
        '''
        
        net_in = self.net_in(x_in)
        net_act = self.net_act(net_in)
        
        return net_act
        
        

    def net_in(self, x_in):
        '''Computes the net_in for neurons in the current recurrent layer. This involves integrating input spikes from:
        - the layer below (`x_in` — e.g. the Input Layer)
        - recurrent connections in the current layer

        Parameters:
        -----------
        x_in: ndarray. shape=(num_neurons_prev_layer,). Spikes produced by the layer below (e.g. the Input Layer).

        Returns:
        -----------
        ndarray. shape=(num_neurons,). Net input of all neurons in the current layer.
        '''
       
    
        
        net_in = x_in @ self.in_rec_wts + self.spikes@self.rec_rec_wts
        
        
        return net_in

    def net_act(self, net_in):
        '''Computes the activation of the neurons in the layer

        Parameters:
        -----------
        net_in: ndarray. shape=(num_neurons,). Net input of all neurons in the current layer.

        Returns:
        -----------
        ndarray of bool. shape=(num_neurons,). Spikes produced all neurons in the current layer at the current time step.

        TODO: Copy-paste your net_act method from lif_neurons.py. You should not need to make any substantial changes except:
            1. Assume that we always use winner-take-all behavior (no need for `is_wta` parameter anymore).
            2. Assume that we always use the adaptive thresholds (no need for `do_thres` parameter anymore).
            3. Only change the adaptive thresholds in any way if we are in learning/training mode.
            4. Adapt any parameter names to your project JSON config file convention for the recurrent layer.
        '''
        
        #compute voltage
        self.v = (self.params["v_rest"] + (self.v - self.params["v_rest"])*self.params["voltage_decay_rate"])
    
        # get index of all avaible neurons
        re_count = (self.refrac_count == 0)
        
        # integrate net_in for avaible neurons; for rest decrement refrac count by 1
        self.v[re_count] = self.v[re_count] + net_in[re_count]
        
        self.refrac_count[~re_count] -= 1
       
       # determine what neurons spiked
        self.spikes = (self.v >= self.params["spike_thre"] +self.adaptive_thres)
            
        self.v[self.spikes] = self.params["v_reset"]
        self.refrac_count[self.spikes] = self.params["refrac_period"]
        if np.sum(self.spikes) > 1:
            spiking_neurons = np.nonzero(self.spikes)[0]
            winner = np.random.choice(spiking_neurons, size=1)
            self.spikes[:] = False
            self.spikes[winner] = True
            
        self.trace[self.spikes] = 1 
        self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
        if self.is_learning:
            self.adaptive_thres[~self.spikes] = self.adaptive_thres[~self.spikes] * self.params["adaptive_decay_rate"]
            self.adaptive_thres[self.spikes] = self.adaptive_thres[self.spikes] + self.params["adaptive_threshold"]
            
     
        '''     
        if self.is_learning:
            self.spikes = (self.v >= self.params["spike_thre"] +self.adaptive_thres)
            
            self.v[self.spikes] = self.params["v_reset"]
            self.refrac_count[self.spikes] = self.params["refrac_period"]
            if np.sum(self.spikes) > 1:
                spiking_neurons = np.nonzero(self.spikes)[0]
                winner = np.random.choice(spiking_neurons, size=1)
                self.spikes[:] = False
                self.spikes[winner] = True
                
            self.adaptive_thres[self.spikes] = self.adaptive_thres[self.spikes] + self.params["adaptive_threshold"]
            self.trace[self.spikes] = 1
            
            self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
            self.adaptive_thres[~self.spikes] = self.adaptive_thres[~self.spikes] * self.params["adaptive_decay_rate"]
        else: 
            self.spikes = (self.v >= self.params["spike_thre"] +self.adaptive_thres)
            self.v[self.spikes] = self.params["v_reset"]
            self.refrac_count[self.spikes] = self.params["refrac_period"]
            if np.sum(self.spikes) > 1:
                spiking_neurons = np.nonzero(self.spikes)[0]
                winner = np.random.choice(spiking_neurons, size=1)
                self.spikes[:] = False
                self.spikes[winner] = True
            #self.adaptive_thres[spikes] = self.adaptive_thres[spikes] + self.params["adaptive_threshold"]
            self.trace[self.spikes] = 1
           
            self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
            #self.adaptive_thres[~spikes] = self.adaptive_thres[~spikes] * self.params["adaptive_decay_rate"]
        '''
        
        return self.spikes

    def update_wts(self, x_in, trace_in):
        '''Update the Input -> Recurrent layer wts according to the STDP learning rule

        Parameters:
        -----------
        x_in: ndarray. shape=(num_neurons_prev_layer,). Spikes in the layer below the current layer (i.e. Input Layer).
        trace_in: ndarray. shape=(num_neurons_prev_layer,). Trace in the layer below the current layer (i.e. Input Layer).

        TODO:
        1. If we are not in learning mode, get out.
        2. Use a combination of the previous layer spikes and trace and the current layer spikes and trace to compute the
        STDP weight update rule.
        3. Clip the weights so that any weights less than 0 are set to 0 and any wts bigger than 1 are set to 1.
        '''
        if not self.is_learning:
            return
        
        d_pos =self.params["wts_update_+"]*(trace_in[:, np.newaxis]) @ (self.spikes[np.newaxis, :]) 
       
        d_neg = self.params["wts_update_-"]*(x_in[:, np.newaxis]) @ (self.trace[np.newaxis, :]) 

        self.in_rec_wts = self.in_rec_wts  + d_pos
        self.in_rec_wts  = self.in_rec_wts  - d_neg
        
        self.in_rec_wts [self.in_rec_wts >1] = 1
        self.in_rec_wts [self.in_rec_wts <0] = 0
        
        self.in_rec_wts = self.in_rec_wts 

    def normalize_wts(self, eps=1e-10):
        '''Normalize the Input -> Recurrent layer wts according to the normalization equation.

        Parameters:
        -----------
        eps: float. Small constant to prevent division by 0 in the wt normalization.

        TODO:
        1. If we are not in learning mode, get out.
        2. Set the input -> recurrent weights to themselves divided by the sum of the absolute value of the weights
        coming in from the previous layer. Multiply these normalized weights by the weight scale parameter, which has 
        default value of 78.4. 

        NOTE: In cases where the sum of the absolute weights coming in are 0, divide by 1 instead.
        '''
        # sum over axis 0
        if not self.is_learning:
            return
        
        sum_wts = np.sum(np.abs(self.in_rec_wts), axis=0)
      
        if sum_wts.all() == 0:
            
            self.in_rec_wts = (self.in_rec_wts/(1 + eps) + eps) * self.params["norm_wts"]
            
        else:
            self.in_rec_wts = (self.in_rec_wts/((np.sum(np.abs(self.in_rec_wts), axis=0))) + eps) * self.params["norm_wts"]


class SpikingNet:
    '''A spiking neural network with an input layer and one or more recurrent spiking layers of LIF neurons.
    
    The architecture is:
        Input Layer -> Recurrent Layer
    '''
    def __init__(self, all_params):
        '''SpikingNet constructor

        Parameters:
        -----------
        all_params: Python dictionary. Holds parameters from the JSON config file for ALL layers.

        TODO: Create input and recurrent layers, set them as instance variables
        '''
        self.inputLayer = InputLayer(all_params['input_layer'])
        self.rec_layer = RecurrentLayer(self.inputLayer.get_num_units(), all_params["recurrent_layer"])

        pass

    def get_input_layer(self):
        '''Gets the input layer'''
        return self.inputLayer

    def get_recurrent_layer(self):
        '''Gets the recurrent layer'''
        return self.rec_layer

    def get_learned_wts(self):
        '''Get the wts of the recurrent layer.
        
        (Provided for you / should not require modification)
        '''
        rec_layer = self.get_recurrent_layer()
        return rec_layer.get_in_wts().copy()

    def set_learned_wts(self, new_wts):
        '''Replace the wts of the recurrent layer to `new_wts`
        
        (Provided for you / should not require modification)
        '''
        rec_layer = self.get_recurrent_layer()
        return rec_layer.set_in_wts(new_wts)
    
    def reset_state(self):
        '''Resets the state (but not adaptive threshold) in all layers of the net'''
        self.inputLayer.reset_state()
        self.rec_layer.reset_state()
        
        
    
    def reset_adaptive_threshold(self):
        '''Resets the adaptive thresholds in all recurrent layers within the net'''
        
        self.rec_layer.reset_adaptive_threshold()
        
    
    def set_is_learning(self, mode):
        '''Sets boolean field of recurrent layer to indicate whether the layer is currently in training or
        prediction mode.

        Parameters:
        -----------
        mode: bool. True if recurrent layer in training mode, False if recurrent layer in prediction mode.
        '''
        self.rec_layer.set_is_learning(mode)

    def forward(self, x):
        '''Forward pass through all layers of the net for the current time step

        There is a time delay of 1 time step between the output of one layer and the input to the next.
        So:
        - Input layer should process the spike input at the CURRENT time step (`x`).
        - Recurrent layer should process the spikes produced by the input layer on the PREVIOUS time step.

        Parameters:
        -----------
        x: ndarray. shape=(M=num_feats,). Poisson-encoded spike input to the network for the current time step.
        '''
        prev_act_inLayer = self.inputLayer.get_spikes()
        in_layer_act_curr = self.inputLayer.forward(x)
        rec_layer_act = self.rec_layer.forward(prev_act_inLayer)
        
        pass

    def simulate(self, x, plot_sample_voltage=False):
        '''Simulates the network for the current data sample `x`. We are assuming that we will process data SGD-style
        (one sample at a time). So we assume `x` coming with has N=1 (but the batch dimension squeezed out).

        Parameters:
        -----------
        x: ndarray. shape=(M,). Raw current input sample to process by the net for the current time step.
            NOT Poisson encoded.
        plot_sample_voltage: bool. Do we make a plot showing the voltage record over all time steps for the current sample
            for all neurons in the recurrent layer.
        
        Returns:
        -----------
        ndarray. shape=(T, num_recurrent_units). Record of recurrent layer spikes at each time step.

        TODO:
        1. Encode the current sample as a Poisson spike train.
        2. To make life easier with the indexing, squeeze out the batch dimension of the Poisson encoding so that the
        encoded spike train has shape (T,M).
        3. Setup containers to record the spikes and voltage of all neurons in the RECURRENT layer at all time steps.
        4. For the the current time step, do the forward pass thru the network and update the recurrent layer wts.
        5. After processing all time steps:
            - Normalize the recurrent layer wts
            - Reset the state of each layer.
        '''
        pos_encode = self.inputLayer.poisson_encode(x).squeeze()
       
        spike_rec = np.zeros([self.inputLayer.params["num_time_steps"]+1, self.rec_layer.params["num_neurons"]], dtype=bool)
        v_rec = np.zeros([self.inputLayer.params["num_time_steps"]+1, self.rec_layer.params["num_neurons"]])
        
        for t in range(1, self.inputLayer.params["num_time_steps"]+1):
            self.forward(pos_encode[t-1, :])
            spike_rec[t] = self.rec_layer.get_spikes()
            v_rec[t] = self.rec_layer.get_v()
            self.rec_layer.update_wts(self.inputLayer.get_spikes(), self.inputLayer.get_trace())
        # Put me after processing all time steps
        if plot_sample_voltage:
            fig = plt.figure(num=2)
            clear_output(wait=True)
            plot_voltage(v_rec)
            plt.pause(0.1)
            fig.canvas.draw()
            
        self.rec_layer.normalize_wts()
        self.reset_state()
        #we are offset by 1 here so we ignore first element as it contains no information
        return spike_rec[1:]
        '''  
         #1. 2. poisson encode input squeezing implicitly here
        M = x.shape
        T = self.params["num_time_steps"]
        N_r = self.rec_layer.get_num_units()
        N_i = self.inputLayer.get_num_units()
        x_a = np.reshape(x/1000, (1,M))
        randoms = np.random.uniform(0,1,(T,M)) 
        poisson =  (0 >= (randoms - x_a)).astype(int)

        #3. spike and voltage for recurrent layer
        spikes = np.zeros((N_r,T,M))
        voltage = np.zeros((N_r,T,M))

        #NEED FOR LOOP FOR TIME HERE?
        for t in range(T):
            #compute forward pass
            self.forward(poisson)
            #update recurrent weights
            self.rec_layer.update_wts(self.inputLayer.get_spikes(),self.inputLayer.get_trace)
        '''

    def accuracy(self, y_pred, y_true):
        '''Computes the accuracy between the predicted labels (`y_pred`) and true labels (`y_true`).

        Parameters:
        -----------
        y_pred: ndarray. shape=(N,). Int coded predicted label of each data sample.
        y_true: ndarray. shape=(N,). Int coded true label of each data sample.

        Returns:
        -----------
        float. The accuracy.
        '''
        return (np.sum(y_pred == y_true)/y_pred.shape[0])


    def assign_neurons_to_classes(self, spike_record, y, num_classes=10):
        '''Determine which class each neuron "votes for" based on largest avg spike rate across samples of each class
        Want: Average spike rate of each neuron to samples of each class.

        (This is provided for you / no modication should be necessary)

        Parameters:
        -----------
        spike_record: ndarray. shape=(N, T, H=num_recurrent_neurons). Spikes produced by all H recurrent layer neurons
            across all T time steps and N samples.
        y: ndarray. shape=(N,). Int coded true label of each data sample.
        num_classes: int. Number of unique classes in the dataset.

        Returns:
        -----------
        ndarray. shape=(H,). Class association of each recurrent layer neuron.
        '''
        _, _, num_neurons = spike_record.shape
        # Collapse across time. shape=(N, num_neurons)
        total_spikes = spike_record.sum(axis=1)
        # Container to hold votes for each class
        votes = np.zeros([num_classes, num_neurons])

        # Get votes for each class
        for c in range(num_classes):
            # Find indices where the sample class matches c, the current class
            inds = np.nonzero(y == c)[0]

            if len(inds) > 0:
                # Select the spikes produced by all neurons to samples of class c and avg across samples of class c
                votes[c] = np.mean(total_spikes[inds], axis=0)
        # The class assignment of each neuron is the class that garners the highest mean spikes
        neuron_labels = np.argmax(votes, axis=0)
        return neuron_labels

    def predict_class(self, spike_record, neuron_class_assignments, num_classes=10):
        '''Predict the class label of each data sample through a vote conducted between recurrent layer neurons that
        represent different classes. For each sample, we collect the total number of spikes produced by the subset of neurons
        associated with the C classes. The predicted class is the index of class association of the group of neurons that
        achieve the highst average number of spikes.

        Example: C=2 classes, N=3 samples, H=6 neurons
            neuron_class_assignments: [1, 0, 1, 0, 0, 0]
            
            spikes: [[1, 1, 2],  # Neuron 1
                     [7, 6, 7],  # Neuron 2
                     [0, 5, 1],  # Neuron 3
                     [0, 1, 0],  # Neuron 4
                     [0, 1, 7],  # Neuron 5
                     [2, 1, 1]]  # Neuron 6
            
            votes:  [[9,  9, 15]   # class 0
                     [1,  6,  3]]  # class 1

            avg votes:  [[9/4,  9/4, 15/4]    # class 0
                         [1/2,  6/2,  3/2]]  # class 1
            
            y_pred: [0, 1, 0]

        Parameters:
        -----------
        spike_record: ndarray. shape=(N, T, H=num_recurrent_neurons). Spikes produced by all H recurrent layer neurons
            across all T time steps and N samples.
        neuron_class_assignments: ndarray. shape=(H,). Class association of each recurrent layer neuron.
        num_classes: int. Number of unique classes in the dataset.

        Returns:
        -----------
        ndarray. shape=(N,). Int coded predicted label of each data sample.

        NOTE:
        - The structure of this method is VERY similar to that of the provided `assign_neurons_to_classes` method.
        - It may make indexing easier if you transpose the spike totals (after summing spikes across time) so that the
        working shape of the spikes is (H, N).
        - Multiple loops are fine, but you can get away with only one.
        '''
        N, _, _ = spike_record.shape
        
        total_spikes = spike_record.sum(axis=1).T
        
        votes = np.zeros([num_classes, spike_record.shape[0]])
        
        for c in range(num_classes):
            # Find indices where the sample class matches c, the current class
            inds = np.nonzero(neuron_class_assignments == c)[0]

            if len(inds) > 0:
                # Select the spikes produced by all neurons to samples of class c and avg across samples of class c
                votes[c] = np.mean(total_spikes[inds], axis=0)

        # The class assignment of each neuron is the class that garners the highest mean spikes
        neuron_labels = np.argmax(votes, axis=0)
        return neuron_labels
    
    def predict(self, x, y=None, neuron_labels=None, num_classes=10):
        '''Predict the class of samples `x`

        (This is provided to you / should not require modification)

        Parameters:
        -----------
        x: ndarray. shape=(N, M). Raw data samples.
        y: ndarray or None. shape=(N,). Data class labels.
            NOTE: Only needed to be passed in if `neuron_labels` is NOT passed in (i.e. we need to associate class labels
            with the data samples `x`).
        neuron_labels: ndarray. shape=(H,). Class association of each recurrent layer neuron.
        num_classes: int: Number of unique classes in dataset.

        Returns:
        -----------
        y_pred: ndarray. shape=(N,). Int coded predicted label of each data sample.
        neuron_labels: ndarray. shape=(H,). int coded association of the H recurrent layer neurons with one of the C classes.

        TODO:
        1. Turn off learning mode in the recurrent layer.
        2. Process each sample and record the spikes produced by H recurrent layer neurons for all N samples and T time
        steps.
        3. If `neuron_labels` has not been passed in, determine the class association of each recurrent layer neuron based
        on the dataset passed in (`x`, `y`).
        4. Use the class associations of each recurrent neuron to determine the predicted labels for every samples in `x`.
        '''
        N = len(x)
        T = self.inputLayer.get_num_time_steps()
        H = self.rec_layer.get_num_units()

        # Set the recurrent layer in prediction mode
        self.set_is_learning(False)

        # Need to store spike record
        spike_rec = np.empty([N, T, H], dtype=bool)

        for i in range(N):
            spike_rec[i] = self.simulate(x[i])
        # Do assignment over dataset
        if neuron_labels is None:
            neuron_labels = self.assign_neurons_to_classes(spike_rec, y, num_classes)
        # Predict classes of validation set
        y_pred = self.predict_class(spike_rec, neuron_labels, num_classes)

        return y_pred, neuron_labels

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1, num_classes=10, print_every=250, val_every=500,
              plot_sample_voltage=False, plot_wts_live=False, plot_wt_rows_cols=(10, 10), plot_pause=0.01):
        '''Train the spiking neural network with the training set (`x_train`, `y_train`).

        Parameters:
        -----------
        x_train: ndarray. shape=(N_train, M). Training set samples.
        y_train: ndarray. shape=(N_train,). Training set labels.
        x_val: ndarray. shape=(N_val, M). Val set samples.
        y_val: ndarray. shape=(N_val,). Val set labels.
        epochs: int. Number of epochs to train.
        num_classes: int: Number of unique classes in dataset.
        print_every: int. Print out training progress after process this many data samples (within epoch).
        val_every: int. Every time this many data samples (within epoch) is processed, compute the validation accuracy.
        plot_sample_voltage: bool. Do we make a plot showing the voltage of every recurrent layer neuron after processing
            each sample?
        plot_wts_live: bool. Do we make a plot showing the input-to-recurrent learned weights after processing each sample?
        plot_wt_rows_cols: tuple. If plotting the weights, plot the weights of this many rows/cols of recurrent neurons.
        plot_pause: float. Time in sec to wait/halt after creating the plot before continuing on with the processing.

        Returns:
        -----------
        train_acc: float. Training set accuracy
        train_neuron_labels: ndarray. shape=(H,). int coded association of the H recurrent layer neurons with one of the
            C classes.

        TODO: 
        0. Reset the state and adaptive thresholds in the network
        1. Set the recurrent layer to be in training mode.
        2. Create a training loop to process one data sample at a time by the network for `epoch` epochs.
        3. Print out training progress every `print_every` samples.
        3. Estimate the validation set accuracy every `val_every` samples. This involves:
            - Predicting the class labels on the validation set.
            - Remember to TURN ON learning/training mode in the recurrent layer AFTER getting the predicted val class
            labels. all val samples have been processed.
            NOTE: To save A LOT of time during the val checks, you can assign the class associations to recurrent neurons
            based on the validation set. In other words, you do not need to determine the class assignments on the
            training set here and then do another pass on the validation set.
        4. Once training is over compute the accuracy on the full training set.
        '''
        # Put me before training loop
        # Create figures (if plotting)
        if plot_wts_live:
            fig = plt.figure(num=1, figsize=(4, 4))
        if plot_sample_voltage:
            fig = plt.figure(num=2, figsize=(4, 4))

        #0. Reset the state and adaptive thresholds in the network
        self.reset_state()
        self.reset_adaptive_threshold()
        #1. Set the recurrent layer to be in training mode.
        self.set_is_learning(True)
        N = len(x_train)
        #2. Create a training loop to process one data sample at a time by the network for `epoch` epochs.
        for i in range(N):
            for epoch in range(epochs):
                self.simulate(x_train[i, :])
                #3. Print out training progress every `print_every` samples.
                if i== 0 or i%print_every == 0:
                    print("iteration = ",i)
                #3. Estimate the validation set accuracy every `val_every` samples.
                if i == 250 or i%val_every == 0:
                    pred_val, _ = self.predict(x_val, y_val)
                    acc_val =self.accuracy(pred_val, y_val)
                    self.set_is_learning(True)
                    
            # Put me after processing the i-th data sample in the training loop
            # Create figures (if plotting)
            if plot_wts_live:
                fig = plt.figure(num=1)
                num_rcs = plot_wt_rows_cols
                clear_output(wait=True)
                draw_grid_image(self.get_learned_wts().T, num_rcs[0], num_rcs[1], title='Learned wts')
                fig.canvas.draw()
                plt.pause(plot_pause)
        
        #4. Once training is over compute the accuracy on the full training set.
        train_pred, neuron_labels = self.predict(x_train,y_train)
        train_acc = self.accuracy(train_pred, y_train)
        
        
        return train_acc, neuron_labels

class SpikingNet2:
    '''A spiking neural network with an input layer and one or more recurrent spiking layers of LIF neurons.
    
    The architecture is:
        Input Layer -> Recurrent Layer -> Recurrent Layer
    '''
    def __init__(self, all_params):
        '''SpikingNet constructor

        Parameters:
        -----------
        all_params: Python dictionary. Holds parameters from the JSON config file for ALL layers.

        TODO: Create input and recurrent layers, set them as instance variables
        '''
        self.inputLayer = InputLayer(all_params['input_layer'])
        self.rec_layer = RecurrentLayer(self.inputLayer.get_num_units(), all_params["recurrent_layer"])
        self.rec_layer2 = RecurrentLayer(self.rec_layer.get_num_units(), all_params["recurrent_layer2"])
        

        pass

    def get_input_layer(self):
        '''Gets the input layer'''
        return self.inputLayer

    def get_recurrent_layer(self):
        '''Gets the recurrent layer'''
        return self.rec_layer
    
    def get_recurrent_layer2(self):
        '''Gets the recurrent layer'''
        return self.rec_layer2

    def get_learned_wts(self):
        '''Get the wts of the recurrent layer.
        
        (Provided for you / should not require modification)
        '''
        rec_layer = self.get_recurrent_layer()
        return rec_layer.get_in_wts().copy()
    
    def get_learned_wts2(self):
        '''Get the wts of the recurrent layer.
        
        (Provided for you / should not require modification)
        '''
        rec_layer2 = self.get_recurrent_layer2()
        return rec_layer2.get_in_wts().copy()

    def set_learned_wts(self, new_wts):
        '''Replace the wts of the recurrent layer to `new_wts`
        
        (Provided for you / should not require modification)
        '''
        rec_layer = self.get_recurrent_layer()
        return rec_layer.set_in_wts(new_wts)
    
    def set_learned_wts2(self, new_wts):
        '''Replace the wts of the recurrent layer to `new_wts`
        
        (Provided for you / should not require modification)
        '''
        rec_layer2 = self.get_recurrent_layer2()
        return rec_layer2.set_in_wts(new_wts)
    
    def reset_state(self):
        '''Resets the state (but not adaptive threshold) in all layers of the net'''
        self.inputLayer.reset_state()
        self.rec_layer.reset_state()
        self.rec_layer2.reset_state()
        
        
    
    def reset_adaptive_threshold(self):
        '''Resets the adaptive thresholds in all recurrent layers within the net'''
        
        self.rec_layer.reset_adaptive_threshold()
        self.rec_layer2.reset_adaptive_threshold()
        
    
    def set_is_learning(self, mode):
        '''Sets boolean field of recurrent layer to indicate whether the layer is currently in training or
        prediction mode.

        Parameters:
        -----------
        mode: bool. True if recurrent layer in training mode, False if recurrent layer in prediction mode.
        '''
        self.rec_layer.set_is_learning(mode)
        self.rec_layer2.set_is_learning(mode)
    

    def forward(self, x):
        '''Forward pass through all layers of the net for the current time step

        There is a time delay of 1 time step between the output of one layer and the input to the next.
        So:
        - Input layer should process the spike input at the CURRENT time step (`x`).
        - Recurrent layer should process the spikes produced by the input layer on the PREVIOUS time step.

        Parameters:
        -----------
        x: ndarray. shape=(M=num_feats,). Poisson-encoded spike input to the network for the current time step.
        '''
        prev_act_inLayer = self.inputLayer.get_spikes()
        in_layer_act_curr = self.inputLayer.forward(x)
        
        prev_act_recLayer = self.rec_layer.get_spikes()
        rec_layer_act = self.rec_layer.forward(prev_act_inLayer)
        
        rec_layer_act2 = self.rec_layer2.forward(prev_act_recLayer)
        
        pass

    def simulate(self, x, plot_sample_voltage=False):
        '''Simulates the network for the current data sample `x`. We are assuming that we will process data SGD-style
        (one sample at a time). So we assume `x` coming with has N=1 (but the batch dimension squeezed out).

        Parameters:
        -----------
        x: ndarray. shape=(M,). Raw current input sample to process by the net for the current time step.
            NOT Poisson encoded.
        plot_sample_voltage: bool. Do we make a plot showing the voltage record over all time steps for the current sample
            for all neurons in the recurrent layer.
        
        Returns:
        -----------
        ndarray. shape=(T, num_recurrent_units). Record of recurrent layer spikes at each time step.

        TODO:
        1. Encode the current sample as a Poisson spike train.
        2. To make life easier with the indexing, squeeze out the batch dimension of the Poisson encoding so that the
        encoded spike train has shape (T,M).
        3. Setup containers to record the spikes and voltage of all neurons in the RECURRENT layer at all time steps.
        4. For the the current time step, do the forward pass thru the network and update the recurrent layer wts.
        5. After processing all time steps:
            - Normalize the recurrent layer wts
            - Reset the state of each layer.
        '''
        pos_encode = self.inputLayer.poisson_encode(x).squeeze()
       
        spike_rec = np.zeros([self.inputLayer.params["num_time_steps"]+1, self.rec_layer2.params["num_neurons"]], dtype=bool)
        v_rec = np.zeros([self.inputLayer.params["num_time_steps"]+1, self.rec_layer2.params["num_neurons"]])
        
        for t in range(1, self.inputLayer.params["num_time_steps"]+1):
            self.forward(pos_encode[t-1, :])
            spike_rec[t] = self.rec_layer2.get_spikes()
            v_rec[t] = self.rec_layer2.get_v()
            self.rec_layer.update_wts(self.inputLayer.get_spikes(), self.inputLayer.get_trace())
            self.rec_layer2.update_wts(self.rec_layer.get_spikes(), self.rec_layer.get_trace())
            print(self.rec_layer2.in_rec_wts[0:5, 0:5], "hhh")
            print(self.rec_layer.in_rec_wts[0:5, 0:5], "ttt")
            
        # Put me after processing all time steps
        if plot_sample_voltage:
            fig = plt.figure(num=2)
            clear_output(wait=True)
            plot_voltage(v_rec)
            plt.pause(0.1)
            fig.canvas.draw()
            
        self.rec_layer.normalize_wts()
        self.rec_layer2.normalize_wts()
        self.reset_state()
        #we are offset by 1 here so we ignore first element as it contains no information
        return spike_rec[1:]
        '''  
         #1. 2. poisson encode input squeezing implicitly here
        M = x.shape
        T = self.params["num_time_steps"]
        N_r = self.rec_layer.get_num_units()
        N_i = self.inputLayer.get_num_units()
        x_a = np.reshape(x/1000, (1,M))
        randoms = np.random.uniform(0,1,(T,M)) 
        poisson =  (0 >= (randoms - x_a)).astype(int)

        #3. spike and voltage for recurrent layer
        spikes = np.zeros((N_r,T,M))
        voltage = np.zeros((N_r,T,M))

        #NEED FOR LOOP FOR TIME HERE?
        for t in range(T):
            #compute forward pass
            self.forward(poisson)
            #update recurrent weights
            self.rec_layer.update_wts(self.inputLayer.get_spikes(),self.inputLayer.get_trace)
        '''

    def accuracy(self, y_pred, y_true):
        '''Computes the accuracy between the predicted labels (`y_pred`) and true labels (`y_true`).

        Parameters:
        -----------
        y_pred: ndarray. shape=(N,). Int coded predicted label of each data sample.
        y_true: ndarray. shape=(N,). Int coded true label of each data sample.

        Returns:
        -----------
        float. The accuracy.
        '''
        return (np.sum(y_pred == y_true)/y_pred.shape[0])


    def assign_neurons_to_classes(self, spike_record, y, num_classes=10):
        '''Determine which class each neuron "votes for" based on largest avg spike rate across samples of each class
        Want: Average spike rate of each neuron to samples of each class.

        (This is provided for you / no modication should be necessary)

        Parameters:
        -----------
        spike_record: ndarray. shape=(N, T, H=num_recurrent_neurons). Spikes produced by all H recurrent layer neurons
            across all T time steps and N samples.
        y: ndarray. shape=(N,). Int coded true label of each data sample.
        num_classes: int. Number of unique classes in the dataset.

        Returns:
        -----------
        ndarray. shape=(H,). Class association of each recurrent layer neuron.
        '''
        _, _, num_neurons = spike_record.shape
        # Collapse across time. shape=(N, num_neurons)
        total_spikes = spike_record.sum(axis=1)
        # Container to hold votes for each class
        votes = np.zeros([num_classes, num_neurons])

        # Get votes for each class
        for c in range(num_classes):
            # Find indices where the sample class matches c, the current class
            inds = np.nonzero(y == c)[0]

            if len(inds) > 0:
                # Select the spikes produced by all neurons to samples of class c and avg across samples of class c
                votes[c] = np.mean(total_spikes[inds], axis=0)
        # The class assignment of each neuron is the class that garners the highest mean spikes
        neuron_labels = np.argmax(votes, axis=0)
        return neuron_labels

    def predict_class(self, spike_record, neuron_class_assignments, num_classes=10):
        '''Predict the class label of each data sample through a vote conducted between recurrent layer neurons that
        represent different classes. For each sample, we collect the total number of spikes produced by the subset of neurons
        associated with the C classes. The predicted class is the index of class association of the group of neurons that
        achieve the highst average number of spikes.

        Example: C=2 classes, N=3 samples, H=6 neurons
            neuron_class_assignments: [1, 0, 1, 0, 0, 0]
            
            spikes: [[1, 1, 2],  # Neuron 1
                     [7, 6, 7],  # Neuron 2
                     [0, 5, 1],  # Neuron 3
                     [0, 1, 0],  # Neuron 4
                     [0, 1, 7],  # Neuron 5
                     [2, 1, 1]]  # Neuron 6
            
            votes:  [[9,  9, 15]   # class 0
                     [1,  6,  3]]  # class 1

            avg votes:  [[9/4,  9/4, 15/4]    # class 0
                         [1/2,  6/2,  3/2]]  # class 1
            
            y_pred: [0, 1, 0]

        Parameters:
        -----------
        spike_record: ndarray. shape=(N, T, H=num_recurrent_neurons). Spikes produced by all H recurrent layer neurons
            across all T time steps and N samples.
        neuron_class_assignments: ndarray. shape=(H,). Class association of each recurrent layer neuron.
        num_classes: int. Number of unique classes in the dataset.

        Returns:
        -----------
        ndarray. shape=(N,). Int coded predicted label of each data sample.

        NOTE:
        - The structure of this method is VERY similar to that of the provided `assign_neurons_to_classes` method.
        - It may make indexing easier if you transpose the spike totals (after summing spikes across time) so that the
        working shape of the spikes is (H, N).
        - Multiple loops are fine, but you can get away with only one.
        '''
        N, _, _ = spike_record.shape
        
        total_spikes = spike_record.sum(axis=1).T
        
        votes = np.zeros([num_classes, spike_record.shape[0]])
        
        for c in range(num_classes):
            # Find indices where the sample class matches c, the current class
            inds = np.nonzero(neuron_class_assignments == c)[0]

            if len(inds) > 0:
                # Select the spikes produced by all neurons to samples of class c and avg across samples of class c
                votes[c] = np.mean(total_spikes[inds], axis=0)

        # The class assignment of each neuron is the class that garners the highest mean spikes
        neuron_labels = np.argmax(votes, axis=0)
        return neuron_labels
    
    def predict(self, x, y=None, neuron_labels=None, num_classes=10):
        '''Predict the class of samples `x`

        (This is provided to you / should not require modification)

        Parameters:
        -----------
        x: ndarray. shape=(N, M). Raw data samples.
        y: ndarray or None. shape=(N,). Data class labels.
            NOTE: Only needed to be passed in if `neuron_labels` is NOT passed in (i.e. we need to associate class labels
            with the data samples `x`).
        neuron_labels: ndarray. shape=(H,). Class association of each recurrent layer neuron.
        num_classes: int: Number of unique classes in dataset.

        Returns:
        -----------
        y_pred: ndarray. shape=(N,). Int coded predicted label of each data sample.
        neuron_labels: ndarray. shape=(H,). int coded association of the H recurrent layer neurons with one of the C classes.

        TODO:
        1. Turn off learning mode in the recurrent layer.
        2. Process each sample and record the spikes produced by H recurrent layer neurons for all N samples and T time
        steps.
        3. If `neuron_labels` has not been passed in, determine the class association of each recurrent layer neuron based
        on the dataset passed in (`x`, `y`).
        4. Use the class associations of each recurrent neuron to determine the predicted labels for every samples in `x`.
        '''
        N = len(x)
        T = self.inputLayer.get_num_time_steps()
        H = self.rec_layer2.get_num_units()

        # Set the recurrent layer in prediction mode
        self.set_is_learning(False)

        # Need to store spike record
        spike_rec = np.empty([N, T, H], dtype=bool)

        for i in range(N):
            spike_rec[i] = self.simulate(x[i])
        # Do assignment over dataset
        if neuron_labels is None:
            neuron_labels = self.assign_neurons_to_classes(spike_rec, y, num_classes)
        # Predict classes of validation set
        y_pred = self.predict_class(spike_rec, neuron_labels, num_classes)

        return y_pred, neuron_labels

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1, num_classes=10, print_every=250, val_every=500,
              plot_sample_voltage=False, plot_wts_live=False, plot_wt_rows_cols=(10, 10), plot_pause=0.01):
        '''Train the spiking neural network with the training set (`x_train`, `y_train`).

        Parameters:
        -----------
        x_train: ndarray. shape=(N_train, M). Training set samples.
        y_train: ndarray. shape=(N_train,). Training set labels.
        x_val: ndarray. shape=(N_val, M). Val set samples.
        y_val: ndarray. shape=(N_val,). Val set labels.
        epochs: int. Number of epochs to train.
        num_classes: int: Number of unique classes in dataset.
        print_every: int. Print out training progress after process this many data samples (within epoch).
        val_every: int. Every time this many data samples (within epoch) is processed, compute the validation accuracy.
        plot_sample_voltage: bool. Do we make a plot showing the voltage of every recurrent layer neuron after processing
            each sample?
        plot_wts_live: bool. Do we make a plot showing the input-to-recurrent learned weights after processing each sample?
        plot_wt_rows_cols: tuple. If plotting the weights, plot the weights of this many rows/cols of recurrent neurons.
        plot_pause: float. Time in sec to wait/halt after creating the plot before continuing on with the processing.

        Returns:
        -----------
        train_acc: float. Training set accuracy
        train_neuron_labels: ndarray. shape=(H,). int coded association of the H recurrent layer neurons with one of the
            C classes.

        TODO: 
        0. Reset the state and adaptive thresholds in the network
        1. Set the recurrent layer to be in training mode.
        2. Create a training loop to process one data sample at a time by the network for `epoch` epochs.
        3. Print out training progress every `print_every` samples.
        3. Estimate the validation set accuracy every `val_every` samples. This involves:
            - Predicting the class labels on the validation set.
            - Remember to TURN ON learning/training mode in the recurrent layer AFTER getting the predicted val class
            labels. all val samples have been processed.
            NOTE: To save A LOT of time during the val checks, you can assign the class associations to recurrent neurons
            based on the validation set. In other words, you do not need to determine the class assignments on the
            training set here and then do another pass on the validation set.
        4. Once training is over compute the accuracy on the full training set.
        '''
        # Put me before training loop
        # Create figures (if plotting)
        if plot_wts_live:
            fig = plt.figure(num=1, figsize=(4, 4))
        if plot_sample_voltage:
            fig = plt.figure(num=2, figsize=(4, 4))

        #0. Reset the state and adaptive thresholds in the network
        self.reset_state()
        self.reset_adaptive_threshold()
        #1. Set the recurrent layer to be in training mode.
        self.set_is_learning(True)
        N = len(x_train)
        #2. Create a training loop to process one data sample at a time by the network for `epoch` epochs.
        for i in range(N):
            for epoch in range(epochs):
                self.simulate(x_train[i, :])
                #3. Print out training progress every `print_every` samples.
                if i== 0 or i%print_every == 0:
                    print("iteration = ",i)
                #3. Estimate the validation set accuracy every `val_every` samples.
                if i == 250 or i%val_every == 0:
                    pred_val, _ = self.predict(x_val, y_val)
                    acc_val =self.accuracy(pred_val, y_val)
                    self.set_is_learning(True)
                    
            # Put me after processing the i-th data sample in the training loop
            # Create figures (if plotting)
            if plot_wts_live:
                fig = plt.figure(num=1)
                num_rcs = plot_wt_rows_cols
                clear_output(wait=True)
                draw_grid_image(self.get_learned_wts().T, num_rcs[0], num_rcs[1], title='Learned wts')
                fig.canvas.draw()
                plt.pause(plot_pause)
        
        #4. Once training is over compute the accuracy on the full training set.
        train_pred, neuron_labels = self.predict(x_train,y_train)
        train_acc = self.accuracy(train_pred, y_train)
        
        
        return train_acc, neuron_labels

