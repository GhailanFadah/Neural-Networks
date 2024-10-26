'''word2vec.py
Skip-gram Word2vec neural network implemented in TensorFlow
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 2: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import numpy as np
import tensorflow as tf


class Skipgram:
    '''A Skip-gram neural network (word2vec approach) that takes target words as data samples and learns to predict the
    surrounding context words ("classes"). The network learns emergent numerical word embedding vectors for the target
    words based on word context.

    The network has an MLP archeciture:
    Input -> Dense (linear act) -> Dense (softmax act)
    '''
    def __init__(self, num_feats, num_hidden, num_classes, wt_stdev=0.1):
        '''Constructor to intialize the network weights and bias. Skip-gram is a 2 layer MLP network (not counting the
        input layer).

        Parameters:
        -----------
        num_feats: int. Num input features (M)
        num_hidden: int. Num of units in the hidden layer (H)
        num_classes: int. Num data classes (C)
        wt_stdev: float. Standard deviation of the Gaussian-distributed weights and bias

        TODO:
        - Create Gaussian-distributed wts and bias tensors for each layer of the network. Remember to wrap your weights
        and bias as tf.Variables for gradient tracking!
        '''
        self.M = num_feats
        self.H = num_hidden
        self.C = num_classes
        self.y_wts =  tf.Variable(tf.random.normal((num_feats, num_hidden), stddev=wt_stdev))
        self.y_b = tf.Variable(tf.random.normal((num_hidden, 1)))
        self.z_wts = tf.Variable(tf.random.normal((num_hidden, num_classes), stddev=wt_stdev))
        self.z_b = tf.Variable(tf.random.normal((num_classes, 1)))
        

    def set_wts(self, y_wts, z_wts):
        '''Replaces the network weights with those passed in as parameters. Used by test code.

        Parameters:
        -----------
        y_wts: tf.Variable. shape=(M, H). New input-to-hidden layer weights.
        z_wts: tf.Variable. shape=(H, C). New hidden-to-output layer weights.
        '''
        self.z_wts = z_wts
        self.y_wts = y_wts

    def set_b(self, y_b, z_b):
        '''Replaces the network biases with those passed in as parameters. Used by test code.

        Parameters:
        -----------
        y_b: tf.Variable. shape=(H,). New hidden layer bias.
        z_b: tf.Variable. shape=(C,). New output layer bias.
        '''
        self.y_b = y_b
        self.z_b = z_b

    def one_hot(self, x, C):
        '''One-hot codes the vector of class labels `y`

        Parameters:
        -----------
        y: ndarray. shape=(B,) int-coded class assignments of training mini-batch. 0,...,numClasses-1
        C: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: tf.constant. tf.float32. shape=(B, C) One-hot coded class assignments.

        Copy-and-paste from Hebbian Learning project.
        '''
        hot = tf.one_hot(x,C)
        return hot
        

    def multi_hot(self, y_ind_list, num_classes):
        '''Multi-hot codes the vector of class labels `y_ind_list`

        Parameters:
        -----------
        y_ind_list: ndarray of ndarrays. len(outer ndarray)=B, len(each inner ndarray)= = Si.
            int-coded class assignments for each sample i in the training mini-batch
            (i.e.) indices of the context words associated with each target word i in the mini-batch.
            NOTE: The number of context words (Si) is NOT THE SAME for each sample (target word) i!
        num_classes int. Number of unique output classes total (i.e. vocab size)

        Returns:
        -----------
        y_multi_h: tf.constant. tf.float32. shape=(B, C) Multi-hot coded class assignments.

        NOTE:
        - Because of the uneven/jagged number of context words for the samples in the mini-batch, it is difficult to use
        only vectorization to implement this. Using one loop here is totally fine!
        - Because you are building a constant tensor, it might be easier to build the mini-batch of multi-hot vectors in
        numpy and then convert to a TensorFlow tensor at the end.
        '''
        
        y_list = []
        
        for i in y_ind_list:
            y_multi_h = np.zeros(num_classes)
            y_multi_h[i] = 1
            y_list.append(y_multi_h)

        
        y_multi_h = tf.constant(y_list, dtype=tf.float32)
        return y_multi_h

    def forward(self, x):
        '''Performs the forward pass through the decoder network with data samples `x`

        Parameters:
        -----------
        x: tf.constant. shape=(B, M). Data samples

        Returns:
        -----------
        net_in: tf.constant. shape=(B, C). net_in in the output layer to every sample in the mini-batch.

        NOTE:
        - You are returning NET_IN rather than the usual NET_ACT here because the Skip-gram loss involves the output
        layer net_in instead of the output layer net_act.
        '''
        
        y_netIn = x @ self.y_wts + tf.transpose(self.y_b)
        
        y_netAct = y_netIn #linear act for hidden layer
        
        z_netIn = y_netAct @ self.z_wts + tf.transpose(self.z_b)
        
        return z_netIn
        

    def loss(self, z_net_in, y_multi_h):
        '''Computes the Skip-gram loss on the current mini-batch using the multi-hot coded class labels `y_multi_h`.

        Parameters:
        -----------
        z_net_in: tf.constant. shape=(B, C). Net input to every sample in the mini-batch in the output layer.
        y_multi_h: tf.constant. tf.float32. shape=(B, C). Multi-hot coded class assignments (i.e. 1s at context word
            indices, 0s elsewhere for each sample).
            For example:
            y_multi_h = [[0 1 0 0]
                         [0 1 0 1]]
            means that sample 0 in the mini-batch has one context word that has index 1 in the vocab. Sample 1 in the
            mini-batch has two context words that have indices 1 and 3 in the vocab.

        Returns:
        -----------
        loss: float. Skip-gram loss averaged over the mini-batch.

        Hints:
        - It may be a good idea to break down the loss equation and implement each component on separate lines of code.
        - Check out https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp
        - To implement the right-most term in the loss equation, it may be helpful to notice that y_multi_h has the same
        shape as z_net_in and y_multi_h has 1s at all the context word positions (yc in the equation) and 0s elsewhere.
        Thus, you can use y_multi_h as a "gate" or a "filter" to extract z_net_in values that are at the appropriate
        context words and at the same time nulling out the remaining z_net_in values that are not desired.
        '''
        
        B = z_net_in.shape[0]
        si = tf.reduce_sum(y_multi_h, axis =1)
    
        first_term =  si * tf.math.log(tf.reduce_sum(tf.exp(z_net_in),axis = 1))
        second_term = tf.reduce_sum(y_multi_h*z_net_in, axis = 1)
        
        answer = 1/B*tf.reduce_sum((first_term - second_term))
        return answer

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

        Copy-and-paste from Hebbian Learning project.
        '''
        return tf.gather(x, indices)
        

    def fit(self, x, y, mini_batch_sz=512, lr=1e-2, n_epochs=400, print_every=100, verbose=True):
        '''Trains the Skip-gram network on the training samples `x` (int-coded target words) and associated labels `y`
        (int-coded context words). Uses the Adam optimizer.

        Parameters:
        -----------
        x: ndarray. ints. shape=(N,). Int-coded data samples in the corpus (target words).
        y: ndarray of ndarrays. len(outer ndarray)=N, len(each inner ndarray)= = Si.
            int-coded class assignments for each sample i in the corpus
            (i.e.) indices of the context words associated with each target word i
            NOTE: The number of context words (Si) is NOT NECESSARILY THE SAME for each sample (target word) i!
        mini_batch_sz: int. Number of samples to include in each mini-batch.
        lr: float. Learning rate used with Adam optimizer.
        n_epochs: int. Train the network for this many epochs
        print_every: int. How often (in epoches) to print out the current epoch and training loss.
        verbose: bool. If set to `False`, there should be no print outs during training. Messages indicating start and
        end of training are fine.


        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.

        TODO:
        Go through the usual motions:
        - Set up Adam optimizer and training loss history tracking container.
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - For each mini-batch, one-hot code target words and multi-hot code context words.
        - Compute forward pass and loss for each mini-batch. Have your Adam optimizer apply the gradients to update each
        wts and bias.
        - Record the average training loss values across all mini-batches.
        '''
        N = len(x)
        train_loss_hist = []
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= lr)
        for epoch in range(n_epochs):
            batch_loss = []
            for batch in range(int(N/mini_batch_sz)):
                
                indices = tf.random.uniform((mini_batch_sz,),0,N-1, dtype = 'int32')
                batch_x = self.extract_at_indices(x, indices)
                batch_y = y[indices]
                one_hot_y_batch = self.multi_hot(batch_y, self.C) 
                one_hot_x = self.one_hot(batch_x, self.C)

                with tf.GradientTape(persistent = True) as tape:
                   
                  
                    net_act = self.forward(one_hot_x)
                    
                   
                    loss = self.loss(net_act,one_hot_y_batch)
                
                dy_wts  = tape.gradient(loss, self.y_wts)
                dy_b = tape.gradient(loss, self.y_b)
                
                dz_wts  = tape.gradient(loss, self.z_wts)
                dz_b = tape.gradient(loss, self.z_b)
                
                optimizer.apply_gradients(zip([dy_wts],[self.y_wts]))
                optimizer.apply_gradients(zip([dy_b], [self.y_b]))
                optimizer.apply_gradients(zip([dz_wts],[self.z_wts]))
                optimizer.apply_gradients(zip([dz_b], [self.z_b]))
                batch_loss.append(loss)
            
            epoch_loss = sum(batch_loss)/len(batch_loss)
            epoch_loss = epoch_loss.numpy()
            train_loss_hist.append(epoch_loss)
            if verbose:
                if epoch == 0 or epoch == n_epochs-1 or epoch%print_every == 0:
                    print("\nepoch: "+str(epoch+"   ")+ " loss: "+str(epoch_loss))
                    
        return train_loss_hist

    def get_word_vector(self, word2ind, word):
        '''Get a single word embedding vector from a trained network

           Parameters:
           -----------
           word2ind: Dictionary. Maps word strings -> int code indices in vocab
           word: Word for which we want to return its word embedding vector

           Returns:
           -----------
           ndarray. Word embedding vector. shape=(embedding_sz,)
           This is the wt vector from the 1st net layer at the index specified by the word's int-code.
        '''
        if word not in word2ind:
            raise ValueError(f'{word} not in word dictionary!')
        else:
            index = word2ind[word]
            
            return self.y_wts[index]

    def get_all_word_vectors(self, word2ind, wordList):
        '''Get all word embedding vectors for the list of words `wordList` from the trained network

           Parameters:
           -----------
           word2ind: Dictionary. Maps word strings -> int code indices in vocab
           wordList: List of strings. Words for which we want to return their word embedding vectors

           Returns:
           -----------
           ndarray. Word embedding vectors. shape=(len(wordList), embedding_sz)
            This is the wt vectors from the 1st net layer at the index specified by each word's int-code.
        '''
        in_list = []
        for word in wordList:
            indx = word2ind[word]
            in_list.append(indx)
      
        
        return self.extract_at_indices(self.y_wts, in_list)
        
        
        
