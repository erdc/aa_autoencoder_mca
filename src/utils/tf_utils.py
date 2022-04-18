#! /usr/bin/env python

import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def gen_batch_ae(X_data, Y_data, X_val, Y_val, batch_size, shuffle_buffer = 100):
    """
    Utility function to create a minibatch generator
    for AA Autoencoder training
    using tensorflow.data.dataset module
    """
    X_data = tf.convert_to_tensor(X_data,dtype=tf.float32)
    Y_data = tf.convert_to_tensor(Y_data,dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val,dtype=tf.float32)
    Y_val = tf.convert_to_tensor(Y_val,dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    train_dataset = train_dataset.batch(batch_size).shuffle(shuffle_buffer)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def gen_batch_lstm(snap, batch_size, Nt, param_list, msg=False):
    """
    Utility function to create a minibatch generator
    for LSTM training
    1. Overlap is not allowed between data
    from different parameter values
    2. The order of appearance for data associated
    to different parameter values is shuffled
    3. For each parameter value, starting point of
    each minibatch is randomly shuffled, but elements
    within each minibatch are kept chronological
    """

    ## Shuffle the order of parameters
    rng = np.random.default_rng()
    shuffled = np.arange(len(param_list))
    rng.shuffle(shuffled)

    batches = []
    batch_ctr = 0

    for indx,val in enumerate(shuffled):
        num_batches = int(np.ceil(Nt[val]//batch_size))
        ## Starting point of dataset for chosen parameter value
        param_id = int(Nt[val]*val)
        ## Shuffle within dataset for each parameter value
        start_id = np.arange(0,Nt[val],batch_size)
        rng.shuffle(start_id)
        for ii in range(num_batches+1):
            batch_ctr +=1
            try:
                end_id = start_id[ii] + np.minimum(batch_size,Nt[val]-start_id[ii])
                if msg:
                    print("Param = %.2f, Batch = %d, [Start, End] = [%d, %d], Size = %d"%(param_list[val],
                                        batch_ctr, start_id[ii], end_id, end_id-start_id[ii]))
                batches.append(snap[param_id+start_id[ii]:param_id+end_id,:])
            except:
                pass
    return batches


# split a multivariate sequence into samples
def split_sequence_multi(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


### Custom loss functions for NN training
@tf.function
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    loss.__name__ = "root_mean_squared_error_loss"

@tf.function
def rel_rms_error_loss(y_true, y_pred):
    err = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    denominator = tf.sqrt(tf.reduce_mean(tf.square(y_true))) + 10 **(-6)
    return  tf.truediv(err, denominator)
    loss.__name__ = "rel_rms_error_loss"

@tf.function
def normalized_mean_squared_error_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    true_norm = tf.reduce_mean(tf.square(y_true)) + 1e-6
    return  tf.truediv(mse, true_norm)
    loss.__name__ = "normalized_mean_squared_error_loss"

@tf.function
def pseudo_Huber_loss(y_true, y_pred, delta=0.25):
    a = tf.reduce_mean(y_pred - y_true)
    return (tf.sqrt(1 + (a/delta)**2) - 1)*delta**2
    loss.__name__ = "pseudo_Huber_loss"

@tf.function
def mean_abs_percentage_error_loss(y_true, y_pred, eta=1e-6):
    return tf.reduce_mean(tf.abs(y_pred - y_true)/(tf.abs(y_true)+eta))
    loss.__name__ = "mean_abs_percentage_error_loss"


@tf.function
def rel_max_absolute_error_loss(y_true, y_pred):
    mxae = tf.math.reduce_max(tf.abs(y_pred - y_true))
    return tf.truediv(mxae,tf.math.reduce_max(tf.abs(y_true)))
    loss.__name__ = "rel_max_absolute_error_loss"

@tf.function
def comb_loss(y_true, y_pred, lb = 0.7, eta=1e-8, delta = 0.5):
    """
    Hybrid loss combining two loss functions,
    (1-lb)*loss_1 + lb*loss_2
    """
    ## MAPE
    loss1 = mean_abs_percentage_error_loss(y_true, y_pred, eta)

    ## pseudo-Huber
    loss2 = pseudo_Huber_loss(y_true, y_pred, delta)

    ## RMSE
    loss3 = root_mean_squared_error_loss(y_true, y_pred)

    ## RRMSE
    loss4 = rel_rms_error_loss(y_true, y_pred)

    ## MaxAbsErr
    loss5 = rel_max_absolute_error_loss(y_true, y_pred)

    ## NormMSE
    loss6 = normalized_mean_squared_error_loss(y_true, y_pred)

    return (1-lb)*loss2 + lb*loss6
    loss.__name__ = "comb_loss"
