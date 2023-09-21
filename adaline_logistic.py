from adaline import Adaline
import numpy as np

class AdalineLogistic(Adaline):
    
    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''

        net_act = 1/(1+np.exp(-net_in))
        return net_act


    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]

        NOTE: Remember to apply the activation function!
        '''
        
        netIn = self.net_input(features)
        netAct = self.activation(netIn)
        
        
        results = np.where(netAct >= 0.5, 1, 0)
        
        
        return results
    
    def compute_loss(self, y, net_act):
        ''' Computes the Cross Entropy loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        L = 1/2 * np.sum((-y*np.log(net_act)-(1-y)*np.log(1-net_act)) ,axis = 0)
        return L
