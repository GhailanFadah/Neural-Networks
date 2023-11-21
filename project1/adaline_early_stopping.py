from adaline import Adaline
import numpy as np

class AdalineEarly(Adaline):
    def fit(self, features, y, n_epochs = 1000, lr = 0.001, tolerance = .02, early_stop = 5):
        ''' Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            Classes corresponding to each input sample (coded -1 or +1).
        n_epochs: int.
            Number of epochs to use for training the network
        lr: float.
            Learning rate used in weight updates during training
        tolerance: float.
            Tolerance of reduction in loss so early stopping does not occur
        early_stopping: int. 
            Number of epochs where tolerance improvement in loss is not met before early stop

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.

        TODO:
        1. Initialize the weights and bias according to a Gaussian distribution centered
            at 0 with standard deviation of 0.01. Remember to initialize the bias in the same way.
        2. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        self.b = np.random.normal(0, 0.01)
        self.wts = np.random.normal(0, 0.01, features.shape[1])

        threshold = None
        count = 0

        for i in range(n_epochs):
            #now we want to undergo the main training loop
            #calculate net_in
            net_in = self.net_input(features)
            #then net act
            net_act = self.activation(net_in)
            errors = y - net_act
            #next we find the gradient
            
            #find loss/acc
            loss = self.compute_loss(y, net_act) #float
            self.loss_history.append(loss)

            acc = self.compute_accuracy(y,self.predict(features))
            self.accuracy_history.append(acc)
                        
            #early stopping
            if i > 5:
                #only check after 5 epochs
                #look at most recent 5 losses
                window = self.loss_history[-5:]
                
                #average for this window
                cur_avg = sum(window)/len(window)
                #initialize threhsold loss
                if threshold == None: 
                    threshold = cur_avg
                #if loss has gone down
                elif threshold-cur_avg < tolerance:
                    #we reset threshold to our new loss and reset count
                    threshold = cur_avg
                    count = 0
                else:
                    #if loss does not pass our tolerance
                    count+=1
                if count >= early_stop:
                    print("Early Stopping at Epoch "+str(i))
                    return self.loss_history, self.accuracy_history 
                
            gradient_b, gradient_w = self.gradient(errors, features)
            #and we want to step in lr*gradient for each feature
            w_step = lr*gradient_w #STEP_j = lr(-sum(error_i)x_ij)
            b_step = lr*gradient_b #STEP_j = lr(-sum(error_i)) where error_i = y_i-netAct_i
            
            self.wts = self.get_wts()-w_step #we negate the gradient to "walk down" the hill
            #UPDATE BIAS
            self.b = self.get_bias()-b_step


    
        return self.loss_history, self.accuracy_history 