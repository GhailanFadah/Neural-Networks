'''single_layer_net.py
Single layer neural networks trained with supervised learning to predict class labels
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 0: TensorFlow and MNIST
Spring 2024

NOTE: Your challenge is to NOT import numpy here!
'''
import tensorflow as tf


class SingleLayerNetwork:
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
        self.wts = None
        self.b = None

    def get_wts(self):
        '''Returns the net wts'''
        pass

    def get_b(self):
        '''Returns the net bias'''
        pass

    def set_wts(self, wts):
        '''Replaces the net weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.Variable. shape=(M, C). New net network weights.
        '''
        pass

    def set_b(self, b):
        '''Replaces the net bias with `b` passed in as a parameter.

        Parameters:
        -----------
        b: tf.Variable. shape=(C,). New net network bias.
        '''
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def fit(self, x, y, x_val=None, y_val=None, batch_size=2048, lr=1e-4, epochs=1, val_every=1, verbose=True):
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

        print(f'Finished training after {e} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist
