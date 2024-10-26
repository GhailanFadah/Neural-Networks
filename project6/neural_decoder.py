'''neural_decoder.py
Linear and nonlinear decoding neural networks trained with supervised learning to predict class labels
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning

NOTE: Your challenge is to NOT import numpy here!
'''
import tensorflow as tf



class NeuralDecoder:
    '''Single layer Neural network trained to predict the class label from data samples
    '''
    def __init__(self, num_features, num_classes, wt_stdev=0.1):
        '''Constructor to intialize the single layer network weights and bias. There is one set of weights and bias.

        Parameters:
        -----------
        num_features: int. Num input features (M)
        num_classes: int. Num data classes (C)
        wt_stdev: float. Standard deviation of the Gaussian-distributed weights and bias

        NOTE: Remember to wrap your weights and bias as tf.Variables for gradient tracking!
        '''
        # Change/set these


        self.num_features = num_features
        self.num_classes = num_classes
        self.wt_stdev = wt_stdev
        
        self.wts = tf.Variable(tf.random.normal((num_features, num_classes),stddev = self.wt_stdev))
        self.b = tf.Variable(tf.random.normal((num_classes,1),stddev = self.wt_stdev))

    def get_wts(self):
        '''Returns the net wts'''
        
        return self.wts
        

    def get_b(self):
        '''Returns the net bias'''
        
        return self.b
        

    def set_wts(self, wts):
        '''Replaces the net weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.Variable. shape=(M, C). New net network weights.
        '''
        self.wts = wts
        

    def set_b(self, b):
        '''Replaces the net bias with `b` passed in as a parameter.

        Parameters:
        -----------
        b: tf.Variable. shape=(C,). New net network bias.
        '''
        
        self.b = b
        

    def one_hot(self, y, C, off_value=0):
        '''One-hot codes the vector of class labels `y`

        Parameters:
        -----------
        y: tf.constant. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total
        off_value: int. The "off" value that represents all other values in each sample's one-hot vector that is not 1.

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.
            e.g. if off_value=-1, y=[1, 0], and C=3, the one-hot vector would be:
            [[-1., 1., -1.], [1., -1., -1.]]
        '''
        
        
        hot = tf.one_hot(y,C, off_value=off_value)
        
        return hot

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
        

    def forward(self, x):
        '''Performs the forward pass through the single layer network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.

        NOTE: Subclasses should implement this (do not implement this method here).
        '''



        pass

    def loss(self, yh, net_act):
        '''Computes the loss on the current mini-batch using the one-hot coded class labels `yh` and `net_act`.

        Parameters:
        -----------
        yh: tf.constant. tf.float32. shape=(B, C). One-hot coded class assignments.
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.

        Returns:
        -----------
        loss: float. Loss computed over the mini-batch.

        NOTE: Subclasses should implement this (do not implement this method here).
        '''
        pass

    def predict(self, x, net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `net_act`. If `net_act` is not passed in,
        the method should compute it in order to perform the prediction.

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples
        net_act: tf.constant. shape=(B, C) or None. Network activation.

        Returns:
        -----------
        y_preds: tf.constant. shape=(B,). int-coded predicted class for each sample in the mini-batch.
        '''
        
        if net_act is None:
            net_act = self.forward(x)

        y_preds = tf.math.argmax(net_act, axis = 1)
        return y_preds

    def extract_at_indices(self, x, indices):
        '''Returns the samples in `x` that have indices `indices` to form a mini-batch.

        Parameters:
        -----------
        x: tf.constant. shape=(N, ...). Data samples or labels
        indices: tf.constant. tf.int32, shape=(B,), Indices of samples to return from `x` to form a mini-batch.
            Indices may be in any order, there may be duplicate indices, and B may be less than N (i.e. a mini-batch).
            For example indices could be [0, 1, 2], [2, 2, 1], or [2, 1].
            In the case of [2, 1] this method would samples with index 2 and index 1 (in that order).

        Returns:
        -----------
        tf.constant. shape=(B, ...). Value extracted from `x` whose sample indices are `indices`.

        Hint: Check out a TF function used in Task 1 of this project. Also see TF tutorial from last semester
        (end of final notebook) for example usage.
        '''

        return tf.gather(x, indices)

    def fit(self, x, y, x_val=None, y_val=None, batch_size=2048, lr=1e-4, max_epochs=1000,patience = 3, val_every=1, verbose=True, early_stopping = True):
        '''Trains the single layer decoder on the training samples `x` (and associated int-coded labels `y`) using the Adam
        optimizer. Mac users can use the "legacy" Adam optimizer.

        Parameters:
        -----------
        x: tf.constant. tf.float32. shape=(N, M). Data samples.
        y: tf.constant. tf.int64. shape=(N,). int-coded class labels
        x_val: tf.constant. tf.float32. shape=(N_val, M). Validation set samples.
        y_val: tf.constant. tf.int64. shape=(N_val,). int-coded validation set class labels.
        batch_size: int. Number of samples to include in each mini-batch.
        lr: float. Learning rate used with Adam optimizer.
        epochs: int. Network should train for this many epochs.
        val_every: int. How often (in epoches) to compute validation set accuracy, loss, and print out training progress
            (current epoch, training loss, val loss, val acc).
        verbose: bool. If set to `False`, there should be no print outs during training. Messages indicating start and
            end of training are fine.


        Returns:
        -----------
        train_loss_hist: Python list of floats. len=epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=epochs/val_freq.
            Loss computed on the validation set every time it is checked (`val_freq`).
        val_acc_hist: Python list of floats. len=epochs/val_freq.
            Accuracy computed on the validation set every time it is checked (`val_freq`).

        TODO:
        Go through the usual motions:
        - Set up Adam optimizer and training+validation loss history tracking containers.
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass and loss for each mini-batch. Have your Adam optimizer apply the gradients to update the
        wts and bias.
        - Record the average training loss values across all mini-batches in each epoch.
        - If we're on the first, max, or an appropriate epoch, check the validation set accuracy and loss.
            - On epochs that you compute the validation accuracy and loss, print out:
            (current epoch, training loss, val loss, val acc).
        '''
        N = len(x)

        #updated to legacy version along with lr
        
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []
        recent_val_losses = []
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= lr)

        for epoch in range(max_epochs):
            #get batch with replacement
            batch_loss = []
            for batch in range(int(N/batch_size)):
                
                indices = tf.random.uniform((batch_size,),0,N-1, dtype = 'int32')
                batch_x = self.extract_at_indices(x, indices)
                batch_y = self.extract_at_indices(y, indices)
                one_hot_y_batch = self.one_hot(batch_y, self.num_classes) 

                with tf.GradientTape(persistent = True) as tape:
                   
                   # needed to transpose self.b
                    net_act = self.forward(batch_x)
                    
                    # changed to one hot version of batch_y
                    loss = self.loss(one_hot_y_batch, net_act)
                
                d_wts  = tape.gradient(loss, self.wts)
                # changed from self.bias to self.b
                d_b = tape.gradient(loss, self.b)
                optimizer.apply_gradients(zip([d_wts],[self.wts]))
                optimizer.apply_gradients(zip([d_b], [self.b]))
                batch_loss.append(loss)
                
            epoch_loss = sum(batch_loss)/len(batch_loss)
            train_loss_hist.append(epoch_loss)
            if epoch == 0 or epoch == max_epochs-1 or epoch%val_every == 0:
                #get val loss 
                predicted = self.predict(x_val)
                acc = self.accuracy(y_val,predicted)
                #updated to train_loss_hist instead of epoch_loss
                val_loss = self.loss(self.one_hot(y_val,self.num_classes), self.forward(x_val))
                recent_val_losses, stop = self.early_stopping(recent_val_losses, val_loss, patience)
                val_acc_hist.append(acc)
                val_loss_hist.append(val_loss)
                if verbose:
                
                    print("\nepoch: "+str(epoch)+"\nvalidation accuracy: "+str(acc)+"\nvalidation loss: "+str(tf.get_static_value(val_loss)))
                    print("lr: "+str(lr))
                if stop and early_stopping: 
                    print("Early stopping initiated for patience = "+str(patience))
                    break

                

        print(f'Finished training after {epoch+1} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist
    
    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

        - When training begins, the recent history of validation loss values `recent_val_losses` is empty (i.e. `[]`).
        When we have fewer entries in `recent_val_losses` than the `patience`, then we just insert the current val loss.
        - The length of `recent_val_losses` should not exceed `patience` (only the most recent `patience` loss values
        are considered).
        - The recent history of validation loss values (`recent_val_losses`) is assumed to be a "rolling list" or queue.
        Remove the oldest loss value and insert the current validation loss into the list. You may keep track of the
        full history of validation loss values during training, but maintain a separate list in `fit()` for this.

        Conditions that determine whether to stop training early:
        - We never stop early when the number of validation loss values in the recent history list is less than patience
        (training is just starting out).
        - We stop early when the OLDEST rolling validation loss (`curr_val_loss`) is smaller than all recent validation
        loss values. IMPORTANT: Assume that `curr_val_loss` IS one of the recent loss values — so the oldest loss value
        should be compared with `patience`-1 other more recent loss values.

        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience` (inclusive).
        curr_val_loss: float. The loss computed on the validation set on the current training epoch.
        patience: int. The patience: how many recent loss values computed on the validation set we should consider when
            deciding whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience` (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool. Should we stop training based on the recent validation loss values and the patience value?

        NOTE:
        - This method can be concisely implemented entirely with regular Python (TensorFlow/Numpy not needed).
        - It may be helpful to think of `recent_val_losses` as a queue: the current loss value always gets inserted
        either at the beginning or end. The oldest value is then always on the other end of the list.
        '''
        stop = False
        if len(recent_val_losses) == patience:
            #if our recent losses is at least patience
            #we need to check 
            popped = recent_val_losses.pop(0)
            recent_val_losses.append(curr_val_loss)
            counter = 0
            for x in range(len(recent_val_losses)-1):
                if recent_val_losses[0] < recent_val_losses[x+1]:
                    counter +=1
                    if counter == len(recent_val_losses)-1:
                        stop = True
                        
                    
        else:
            recent_val_losses.append(curr_val_loss)
        
        return recent_val_losses, stop
    
class SoftmaxDecoder(NeuralDecoder):
    
    def forward(self,x):
        '''
        Returns netAct through single layer softmax

            Parameters:
            -----------
            x: tf.constant. shape=(N, ...). Data samples 

            Returns:
            -----------
            tf.constant. shape=(N,C). probabilities for each class
        
        '''
        
        net_in = x @ self.wts + tf.transpose(self.b)
        
        net_act = tf.nn.softmax(net_in)

            

        return net_act

    
    def loss(self, yh, net_act):
        '''
        Cross Entropy loss (one hot version)

        Parameters: 
        ------------
        yh: tf.constant true classes (one hot encoded)
        net_act: net_act values passed through 

        Returns
        ----------
        loss: loss over current iteration based on yh and net_acts
        grads: gradient of the loss over this iteration
        '''

        B = net_act.shape[0]
        
        class_sum = tf.math.reduce_sum(tf.math.multiply(tf.cast(yh,'float'), tf.cast(tf.math.log(net_act),'float')), axis = 1)
        batch_sum = tf.math.reduce_sum(class_sum, axis = 0)

        loss = -1/B * batch_sum
    

        return loss
        
    
class NonlinearDecoder(NeuralDecoder):

    def __init__(self, num_features, num_classes, wt_stdev = 0.1, beta = 0.005, loss_exp = 5):
        super().__init__(num_features, num_classes, wt_stdev)
        self.beta = beta 
        self.loss_exp = loss_exp
        

    def one_hot(self, y, C):
        '''One-hot codes the vector of class labels `y` with -1 as off value

        Parameters:
        -----------
        y: tf.constant. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.
            e.g. if y=[1, 0], and C=3, the one-hot vector would be:
            [[-1., 1., -1.], [1., -1., -1.]]
        '''
        hot = tf.one_hot(y,C, off_value = -1)
        return hot
    
    def forward(self,x):
        '''Performs the forward pass through the single layer network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_act: tf.constant. shape=(B, C). Network activation to every sample in the mini-batch.
        '''

        net_in = x @ self.wts + tf.transpose(self.b)
        
        net_act = tf.nn.tanh(self.beta * net_in)

        return net_act

    def loss(self, yh, net_act):

        '''
        Krotov Hopfield loss (one hot version)

        Parameters: 
        ------------
        yh: tf.constant true classes (one hot encoded with off value = -1)
        net_act: net_act values passed through 

        Returns
        ----------
        loss: loss over current iteration based on yh and net_acts
        grads: gradient of the loss over this iteration
        '''

        to_sum = tf.math.pow(tf.abs(tf.cast(yh, tf.float32)-net_act),self.loss_exp)
        loss = tf.reduce_sum(to_sum)
        return loss


