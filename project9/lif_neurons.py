'''lif_neurons.py
YOUR NAMES HERE
CS 443: Bio-Inspired Machine Learning
Project 4: Spiking Neural Networks
'''
import numpy as np


class LIFNeuron:
    '''A single leaky integrate-and-fire (LIF) neuron that can enter a refractory period'''
    def __init__(self, v_rest=-65):
        '''Constructor

        Parameters:
        -----------
        v_rest: float. Neuron's resting voltage (in mV).

        TODO: Set instance variable for the resting voltage.
        '''
        self.v_rest = v_rest

        # Neuron's voltage/membrane potential (mV). float.
        self.v = None
        # How many more time steps the neuron must remain in a refractory period. int.
        self.refrac_count = None  # NEW

    def initialize_state(self):
        '''Initializes the LIF neuron's state:
        - its voltage/membrane potential set to the resting voltage
        - its refractory count set to 0 (i.e. neuron not currently in a refractory period). 
        '''
        self.v = self.v_rest
        self.refrac_count = 0
        pass

    def simulate(self, net_in, num_steps=350, v_reset=-63, v_decay_rate=0.99, v_thres=-60, refrac_len=50):
        '''Simulates the LIF neuron dynamics for `num_steps` during which it integrates the `net_in` at each time step.

        Parameters:
        -----------
        net_in: ndarray. shape=(num_steps,). Input spike train to the neuron at each time step.
        num_steps: int. Number of time steps / length of the simulation in msec.
        v_decay_rate: float. Rate at which the neuron's voltage decays/decreases in the absence of any input.
        v_thres: float. The voltage threshold (in mV).
            When the neuron's voltage reaches equals or exceeds this level, the neuron spikes.
        refrac_len: int. Number of time steps (msec) that the neuron "sits out for" after it spikes and cannot spike again.

        Returns:
        -----------
        spike_rec: ndarray of bool. shape=(num_steps+1,). Whether the neuron spiked at each time step.
        v_rec: ndarray. shape=(num_steps+1,). Record of the neuron's voltage at every time.

        TODO:
        1. Initialize the neuron's state.
        2. Initialize spike and voltage records. Assume that the neuron never spikes at t=0 and it starts at its resting
        voltage.
        3. Update the voltage at each time step.
        4. If the neuron is not in a refractory period, integrate the current input, otherwise just decrement the
        neuron's refractory count.
        5. If the neuron spikes, set the voltage to its **reset** voltage (not reseting voltage), record the spike, and
        start the refractory period.
        '''
        self.initialize_state()
        
        # initialize spkie and v records
        v_rec = np.zeros(num_steps+1)
        v_rec[0] = self.v_rest
        spike_rec = np.zeros(num_steps+1, dtype=bool)
        
        for t in range(1, num_steps+1):
            # Do voltage decay
            v_rec[t] = self.v_rest + (v_rec[t-1] - self.v_rest)*v_decay_rate
            # Integrate net_in
            if self.refrac_count != 0:
                
                self.refrac_count -= 1
            else:
                
                v_rec[t] = v_rec[t] + net_in[t-1] 
            

                # Check for spike. Did we reach threshold?
                spike_rec[t] = v_rec[t] >= v_thres
                
                # If we spiked, set voltage to reset
                if spike_rec[t]:
                    v_rec[t] = v_reset
                    self.refrac_count = refrac_len

        return spike_rec, v_rec


class LIFLayer:
    '''Layer of multiple leaky integrate-and-fire neurons.'''
    def __init__(self, params):
        '''Constructor
        
        Parameters:
        -----------
        params: Python dictionary. Parameters used to simulate the LIF neurons.

        TODO: Set an instance variable for the parameter dictionary.
        '''
        self.params = params
        

    def initialize_state(self):
        '''Initializes the state of the LIF neurons, which means
        - Setting the voltage of each neuron to the resting voltage.
        - Setting the spikes of each neuron to all False/0.
        - Setting the refractory count of each neuron to 0.
        '''
        num_neurons = self.params["num_neurons"]
        
        
        self.v = self.params["v_rest"]*np.ones(num_neurons)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.refrac_count = np.zeros(num_neurons)
        self.trace = np.zeros(num_neurons)
        self.adaptive_thres = np.zeros(num_neurons)

    def reset_state(self):
        '''Resets the state of the LIF neurons, which means:
        - Resetting the voltage of each neuron to its **resting voltage**.
        - Resetting the spikes of each neuron to all False/0.
        - Resetting the refractory count of each neuron to 0.

        HINT: np.fill may be a helpful function here...
        '''
        num_neurons = self.params["num_neurons"]
        self.v = self.params["v_rest"]*np.ones(num_neurons)
        self.spikes = np.zeros(num_neurons, dtype=bool)
        self.refrac_count = np.zeros(num_neurons)
        self.trace = np.zeros(num_neurons)

    def get_v(self):
        '''Gets the current voltage of each neuron.

        Returns:
        -----------
        ndarray. shape=(num_neurons). The current voltage of each neuron.
        '''
        
        return self.v

    def get_spikes(self):
        '''Gets the current spike status of each neuron.

        Returns:
        -----------
        ndarray of bool. shape=(num_neurons). Whether each neuron is currently spiking.
        '''
        return self.spikes

    def get_refrac_count(self):
        '''Gets the refractory count of each neuron — i.e. how much longer each neuron is in its refractory period.

        Returns:
        -----------
        ndarray. shape=(num_neurons). The refractory count of each neuron.
        '''
        return self.refrac_count

    def get_trace(self):
        '''Gets the current memory trace of each neuron.

        NOTE: Added/implemented later

        Returns:
        -----------
        ndarray. shape=(num_neurons). The current memory trace of each neuron.
        '''
        # ADDED 1
        return self.get_trace()

    def get_adaptive_thres(self):
        '''Gets the adaptive threshold of each neuron.

        NOTE: Added/implemented later

        Returns:
        -----------
        ndarray. shape=(num_neurons). The current adaptive threshold of each neuron.
        '''
        # ADDED 2
        return self.adaptive_thres

    def net_act(self, net_in, do_thres=False, do_wta=False):
        '''Computes the activation of the LIF neurons at the current time step.

        Parameters:
        -----------
        net_in: ndarray. shape=(num_neurons,). Input spike train to each neuron at the CURRENT time step.
        do_thres: bool. Whether to simulate adaptive thresholds (ignore during initial implementation).
        do_wta: bool. Whether to only allow one neuron to spike at a time (ignore during initial implementation).

        TODO:
        1. Update each neuron's voltage.
            - Only allow neurons not in their refractory periods to integrate their net_in.
            - Neurons in their refractory period gets their count decremented.
        2. Figure out which neurons spike at the current time step.
        3. For neurons that spike: reset their voltage to the reset voltage and have them enter their refractory period.
        
        NOTE:
        - This method should have zero loops
        - It is best/cleanest if use only logical indexing — no if/else statements
            (except for those added later for do_thres and do_wta)
        '''
        
        #compute voltage
        self.v = (self.params["v_rest"] + (self.v - self.params["v_rest"])*self.params["v_decay_rate"])
        
        # get index of all avaible neurons
        re_count = (self.refrac_count == 0)
        
        # integrate net_in for avaible neurons; for rest decrement refrac count by 1
        self.v[re_count] = self.v[re_count] + net_in[re_count]
        self.refrac_count[~re_count] -= 1
       
       # determine what neurons spiked
        if do_thres:
            self.spikes = (self.v >= self.params["threshold"] +self.adaptive_thres)
            self.v[self.spikes] = self.params["v_reset"]
            self.refrac_count[self.spikes] = self.params["refrac_period"]
            if do_wta:
                if np.sum(self.spikes) > 1:
                    spiking_neurons = np.nonzero(self.spikes)[0]
                    winner = np.random.choice(spiking_neurons, size=1)
                    self.spikes[:] = False
                    self.spikes[winner] = True
       
        else: 
            self.spikes = (self.v >= self.params["threshold"])
            self.v[self.spikes] = self.params["v_reset"]
            self.refrac_count[self.spikes] = self.params["refrac_period"]
            if do_wta:
                if np.sum(self.spikes) > 1:
                    spiking_neurons = np.nonzero(self.spikes)[0]
                    winner = np.random.choice(spiking_neurons, size=1)
                    self.spikes[:] = False
                    self.spikes[winner] = True
        self.adaptive_thres[self.spikes] = self.adaptive_thres[self.spikes] + self.params["adaptive_threshold"]
        self.trace[self.spikes] = 1
        self.adaptive_thres[~self.spikes] = self.adaptive_thres[~self.spikes] * self.params["adaptive_decay_rate"]
        self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
        
       
        
        

    def simulate(self, net_in, num_steps=350, do_thres=False, do_wta=False):
        '''Simulates the layer of LIF neurons for `num_steps` during which it integrates the `net_in` at each time step.

        Parameters:
        -----------
        net_in: ndarray. shape=(num_steps, num_neurons). Input spike train to each neuron at each time step.
        num_steps: int. Number of time steps / length of the simulation in msec.
        do_thres: bool. Whether to simulate adaptive thresholds (ignore during initial implementation).
        do_wta: bool. Whether to only allow one neuron to spike at a time (ignore during initial implementation).

        Returns:
        -----------
        spike_rec: ndarray of bool. shape=(num_steps+1,). Whether the neuron spiked at each time step.
        trace_rec: ndarray. shape=(num_steps+1,). Record of the neuron's memory trace at every time.
        thres_rec: ndarray. shape=(num_steps+1,). Record of the neuron's adaptive threshold at every time.

        TODO:
        1. Initialize the neuron's state.
        2. Initialize spike, trace, and threshold voltage records. Even though you are not initially adding support for
        memory traces and adaptive thresholds, still initialize all of these arrays and return them.
            Assume that each neuron never spikes at t=0, the traces start at 0, and the adaptive thresholds start at 0.
        3. Compute the net_act at the current time based on the input signal at the previous time step.
        4. Record the spikes at every time step.
        5. Return all 3 record arrays.
        '''

        #initialize state
        self.initialize_state()

        #initialize spike, trace, and thres records
        v_rec = np.zeros([num_steps+1, self.params["num_neurons"]])
        v_rec[0] = self.params["v_rest"]
        
        spike_rec = np.zeros([num_steps+1, self.params["num_neurons"]], dtype=bool)
        trace_rec = np.zeros([num_steps+1, self.params["num_neurons"]])
        thres_rec = np.zeros([num_steps+1, self.params["num_neurons"]])
      
        for t in range(1, num_steps+1):
            net_in_1 = net_in[t-1,:]
           
            self.net_act(net_in_1, do_thres, do_wta)
            
            v_rec[t] = self.v
            spike_rec[t] = self.spikes
            trace_rec[t] = self.trace
            thres_rec[t] = self.adaptive_thres
        
        '''
        
        #for loop over time
        for t in range(1, num_steps+1):
            # decay neuron's voltage
            self.v = self.params["v_rest"] + (self.v - self.params["v_rest"])*self.params["v_decay_rate"]
            
            re_count = (self.refrac_count == 0)
        
            self.v[re_count] = self.v[re_count] + net_in[t-1, re_count]
            self.refrac_count[~re_count] -= 1
        
            if do_thres:
                self.spikes = (self.v >= self.params["threshold"] +self.adaptive_thres)
                self.refrac_count[self.spikes] = self.params["refrac_period"]
                self.v[self.spikes] = self.params["v_rest"]
                if do_wta and np.sum(self.spikes)>1: 
                    #get indices of spikes, then pick one randomly
                    indices = np.argwhere(self.spikes).flatten()
                    random = np.random.randint(0,indices.shape[0])
                    self.spikes = np.zeros((self.params["num_neurons"],)) #might want to do this inplace
                    self.spikes[indices[random]] = 1
                    self.spikes = self.spikes.astype(bool)

                    

                self.adaptive_thres[self.spikes] = self.adaptive_thres[self.spikes] + self.params["adaptive_threshold"]
                self.adaptive_thres[~self.spikes] = self.adaptive_thres[~self.spikes] * self.params["adaptive_decay_rate"]
            else: 
                self.spikes = (self.v >= self.params["threshold"])
                self.refrac_count[self.spikes] = self.params["refrac_period"]
                self.v[self.spikes] = self.params["v_rest"]
                if do_wta and np.sum(self.spikes)>1: 
                    #get indices of spikes, then pick one randomly
                    indices = np.argwhere(self.spikes).flatten()
                    random = np.random.randint(0,indices.shape[0])
                    self.spikes = np.zeros((self.params["num_neurons"],)) #might want to do this inplace
                    self.spikes[indices[random]] = 1
                    self.spikes = self.spikes.astype(bool)
            
            self.trace[self.spikes] = 1
            
            # update
            self.trace[~self.spikes] = self.params["trace_decay_rate"]*self.trace[~self.spikes]
            
            # Record the state of our neurons
            v_rec[t] = self.v
            spike_rec[t] = self.spikes
            trace_rec[t] = self.trace
            thres_rec[t] = self.adaptive_thres
        '''
        return spike_rec, trace_rec, thres_rec

                
            
                
                


                

            
            
