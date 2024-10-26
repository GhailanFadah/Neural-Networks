'''hebb_net.py
Bio-inspired neural network that implements the Hebbian learning rule and competition among neurons in the network
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

from viz import draw_grid_image


class HebbNet:
    '''Single layer bio-inspired neural network in which neurons compete with each other and learning occurs via a
    competitive variant of a Hebbian learning rule (Oja's Rule).

    NOTE: This network should once again be implemented in 100% TensorFlow, except where noted below.
    '''
    def __init__(self, num_features, num_neurons, wt_minmax=(0., 1.), k=6, inhib_value=-0.4,
                 load_wts=False, saved_wts_path='export/wts.npy'):
        '''Hebbian network constructor

        Parameters:
        -----------
        num_features: int. Num input features (M)
        num_neurons: int. Num of neurons in the network (H)
        wt_minmax: tuple of 2 floats. wt_minmax[0] is the min possible wt value when wts initialized. wt_minmax[1] is
            the max possible wt value when wts initialized.
        k: int. In the neural competition that occurs when processing each data sample, the neuron that achieves the kth
            highest net_in value ("neuron came in kth place") is inhibited, which means the kth place neuron gets netAct
            value of `-inhib_value`.
        inhib_value: float. Non-positive number (≤0) that represents the netAct value assigned to the inhibited neuron
            (with the kth highest netAct value).
        load_wts: bool. Whether to load weights previously saved off by the network after successful training.
        saved_wts_path: str. Path from the working project directory where the weights previously saved by the net are
            stored. Used if `load_wts` is True.

        TODO:
        - Create instance variables for the parameters
        - Initialize the wts.
            - If loading wts, set the wts by loading the previously saved .npy wt file. Use `np.load` (this use is allowed)
            - Otherwise, create a uniform random weight tensor between the range `wt_minmax`. shape=(M, H). Should NOT
            be a `tf.Variable` because we are not tracking gradients here.
        '''
        self.saved_wts_path = 'export/wts.npy'
        self.num_features = num_features
        self.num_neurons = num_neurons
        self.k = k
        self.inhib_value = inhib_value
        self.wt_minmax = wt_minmax
        if load_wts:
            self.wts = tf.constant(np.load(saved_wts_path))
            print('Loaded stored wts.')
        else:
            self.wts = tf.constant(tf.random.uniform((num_features, num_neurons), wt_minmax[0], wt_minmax[1]))

    def get_wts(self):
        '''Returns the Hebbian network wts'''
        
        return self.wts
        

    def set_wts(self, wts):
        '''Replaces the Hebbian network weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.constant. shape=(M, H). New Hebbian network weights.
        '''
        self.wts = wts

    def net_in(self, x):
        '''Computes the Hebbian network Dense net_in

        Parameters:
        -----------
        x: ndarray. shape=(B, M)

        Returns:
        -----------
        netIn: tf.constant. shape=(B, H)
        '''
        
        return tf.constant(x@self.wts)
        

    def net_act(self, net_in):
        '''Compute the Hebbian network activation, which is a function that reflects competition among the neurons
        based on their net_in values.

        NetAct (also see notebook):
        - 1 for neuron that achieves highest net_in value to sample i
        - -delta for neuron that achieves kth highest net_in value to sample i
        - 0 for all other neurons

        Parameters:
        -----------
        net_in: tf.constant. shape=(B, H)

        Returns:
        -----------
        netAct: tf.constant. shape=(B, H)

        NOTE:
        - It may be helpful to think of this as a two step process: dealing with the winner and then kth place inhibition
        separately.
        - It may be helpful to think of competition as an assignment operation. But you may not be able to use []
        indexing. Instead, refer to Project 0 and think about the TensorFlow analog of arange indexing.
        += is a valid TensorFlow operator.
        '''
        
        B, H = net_in.shape

        # Get the indices of the neuron with the highest net_in value
        top_vals , top_indices = tf.math.top_k(net_in, self.k)
        
        # Create the indices for scatter_nd
        indices_win = tf.stack([tf.range(B), top_indices[:,0]], axis=1)
        indices_inhib = tf.stack([tf.range(B), top_indices[:,-1]], axis=1)
        # Create a mask to zero out all indices except the top indices
        mask = tf.scatter_nd(indices_win, tf.ones(B, dtype=tf.float32), (B, H))+tf.scatter_nd(indices_inhib, tf.ones(B, dtype=tf.float32)*-self.inhib_value, (B, H))
        
        return mask
       
       

    def update_wts(self, x, net_in, net_act, lr, eps=1e-10):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (competitive Oja's rule).
        After computing the weight change based on the current mini-batch, the weight changes (gradients) are normalized
        by the largest gradient (in absolute value). This has the effect of making the largest weight change equal in
        absolute magnitude to the learning rate `lr`. See notebook for equations.

        Parameters:
        -----------
        net_in: tf.constant. shape=(B, H)
        net_act: tf.constant. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        NOTE:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        - The `keepdims` keyword argument may be convenient here.
        '''

        pos= tf.transpose(x) @ net_act
        neg = self.get_wts()*tf.math.reduce_sum(tf.math.multiply(net_in,net_act), axis = 0, keepdims = True)
        d_wts =pos - neg
        d_wts = d_wts/(tf.math.reduce_max(tf.abs(d_wts))+eps)
        new_wts = self.wts+lr*d_wts
        self.set_wts(new_wts)
        pass

    def fit(self, x, n_epochs=1, mini_batch_sz=128, lr=2e-2, plot_wts_live=False, fig_sz=(9, 9), n_wts_plotted=(10, 10),
            print_every=1, save_wts=True):
        '''Trains the Competitive Hebbian network on the training samples `x` using unsupervised Hebbian learning
        (without classes y!).

        Parameters:
        -----------
        x: tf.constant. dtype=tf.float32. shape=(N, M). Data samples.
        n_epochs: int. Number of epochs to train the network.
        mini_batch_sz: int. Mini-batch size used when training the Hebbian network.
        lr: float. Learning rate used with Hebbian weight update rule
        plot_wts_live: bool. Whether to plot the weights and update throughout training every `print_every` epochs.
        save_wts: bool. Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.

        TODO:
        Very similar workflow to usual:
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True. Using `np.save` is totally fine here.
        '''
        N = len(x)
        
        for e in range(n_epochs):
            for batch in range(int(N/mini_batch_sz)):
                indices = tf.random.uniform((mini_batch_sz,),0,N-1, dtype = 'int32')
                batch_x = tf.gather(x, indices)
                net_in = self.net_in( batch_x)
                net_act = self.net_act(net_in)
                self.update_wts(batch_x, net_in, net_act, lr)
              
            if e == 0 or e == n_epochs-1 or e%print_every == 0:  
                if plot_wts_live:
                    fig = plt.figure(figsize=fig_sz)

                # Put this in your training loop
                if plot_wts_live:
                    draw_grid_image(self.wts.numpy().T, n_wts_plotted[0], n_wts_plotted[1], title=f'Net receptive fields (Epoch {e})')
                    display.clear_output(wait=True)
                    display.display(fig)
                    time.sleep(0.001)
                else:
                    print(f'Starting epoch {e}/{n_epochs}')

        if save_wts:
            print('Saving weights...', end='')
            np.save(self.saved_wts_path, self.get_wts())
            print('Done!')



