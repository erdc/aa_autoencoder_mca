#! /usr/bin/env python

import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
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
        
        ## Shuffle within data for each parameter value
        start_id = np.arange(0,Nt[val],batch_size)
        rng.shuffle(start_id)
        for ii in range(num_batches+1):
            batch_ctr +=1
            try:
                end_id = start_id[ii] + np.minimum(batch_size,Nt[val]-start_id[ii])
                if msg:
                    print("Speed = %d, Batch = %d, [Start, End] = [%d, %d], Size = %d"%(c[val], 
                                        batch_ctr, start_id[ii], end_id, end_id-start_id[ii]))
                batches.append(snap[start_id[ii]:end_id,:])
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
    return array(X), array(y)


### Custom loss functions for NN training
@tf.function
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))) 

@tf.function
def rel_rms_error_loss(y_true, y_pred):
    err = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    denominator = tf.sqrt(tf.reduce_mean(tf.square(y_true))) + 10 **(-6)
    return  tf.truediv(err, denominator)

@tf.function
def normalized_mean_squared_error_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    true_norm = tf.reduce_mean(tf.square(y_true)) + 1e-6
    return  tf.truediv(mse, true_norm)

@tf.function
def pseudo_Huber_loss(y_true, y_pred, delta=0.25):
    a = tf.reduce_mean(y_pred - y_true)
    return (tf.sqrt(1 + (a/delta)**2) - 1)*delta**2
    
@tf.function    
def mean_abs_percentage_error_loss(y_true, y_pred, eta=1e-6):
    return tf.reduce_mean(tf.abs(y_pred - y_true)/(tf.abs(y_true)+eta))

@tf.function
def max_absolute_error_loss(y_true, y_pred):
    return K.max(tf.abs(y_pred - y_true))


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
    loss5 = max_absolute_error_loss(y_true, y_pred)

    ## NormMSE
    loss6 = normalized_mean_squared_error_loss(y_true, y_pred)
    
    return (1-lb)*loss2 + lb*loss6



class AAautoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim, act, size, **kwargs):
        super(AAautoencoder, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.activation = act
        self.hidden_units = size[1:]
        try:
            self.l2_lam = kwargs['l2_lam']
        except:
            self.l2_lam = 1e-6
        
        l2 = regularizers.l2(self.l2_lam)
        encoder_input  = keras.Input(shape=(size[0],))
        encoder_output = keras.layers.Dense(latent_dim,activation='linear')
        decoder_input  = keras.Input(shape=(latent_dim,))
        decoder_output = keras.layers.Dense(size[0],activation='linear')
        
        encoder_layers = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in self.hidden_units]
        shift_layers   = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in np.flip(self.hidden_units)]
        decoder_layers = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in np.flip(self.hidden_units)]
        
        self.encoder = encoder_input
        for layer in encoder_layers:
            self.encoder = layer(self.encoder)
        self.encoder = encoder_output(self.encoder)
        
        self.shift = decoder_input
        for layer in shift_layers:
            self.shift = layer(self.shift)
        self.shift = decoder_output(self.shift)
        
        self.decoder = decoder_input
        for layer in decoder_layers:
            self.decoder = layer(self.decoder)
        self.decoder = decoder_output(self.decoder)


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        return {"hidden_units": self.hidden_units, 
                "activation": self.activation, 
                "latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


