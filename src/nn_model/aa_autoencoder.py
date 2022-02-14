#! /usr/bin/env python

import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


class Encoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        print(list(kwargs.keys()))
        l2 = regularizers.l2(kwargs['regu'])
        self.latent_dim = latent_dim
        self.activation = act
        self.hidden_units = hidden_dim[1:]
        self.input_dim = hidden_dim[0]
        self.input_layer  = keras.Input(shape=(self.input_dim,))
        self.output_layer = keras.layers.Dense(self.latent_dim, activation='linear')
        self.hidden_layers = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in self.hidden_units]

    def call(self, input_features):
        x = self.input_layer(input_features)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)



class Shift(tf.keras.layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name ="shift", **kwargs):
        super(Shift, self).__init__(name=name, **kwargs)

        l2 = regularizers.l2(kwargs['regu'])
        self.activation = act
        self.hidden_units = np.flip(hidden_dim[1:])
        self.input_dim = latent_dim
        self.input_layer  = keras.Input(shape=(self.input_dim,))
        self.output_layer = keras.layers.Dense(self.hidden_units[-1], activation='linear')
        self.hidden_layers = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in self.hidden_units]

    def call(self, latent_code):
        x = self.input_layer(latent_code)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        l2 = regularizers.l2(kwargs['regu'])
        self.activation = act
        self.hidden_units = np.flip(hidden_dim[1:])
        self.input_dim = latent_dim
        self.input_layer  = keras.Input(shape=(self.input_dim,))
        self.output_layer = keras.layers.Dense(self.hidden_units[-1], activation='linear')
        self.hidden_layers = [keras.layers.Dense(u,activation=act,kernel_regularizer=l2) for u in self.hidden_units]

    def call(self, latent_code):
        x = self.input_layer(latent_code)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)



class AAautoencoder(tf.keras.Model):

    def __init__(self, latent_dim, act, hidden_dim, name="autoencoder", **kwargs):
        super(AAautoencoder, self).__init__(name=name, **kwargs)

        try:
            self.l2_lam = kwargs['l2_lam']
        except:
            self.l2_lam = 1e-6

        print(self.l2_lam)
        self.encoder = Encoder(latent_dim, act, hidden_dim,regu=self.l2_lam)
        self.shift = Shift(latent_dim, act, hidden_dim,regu=self.l2_lam)
        self.decoder = Decoder(latent_dim, act, hidden_dim,regu=self.l2_lam)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        pred = self.decoder(encoded)
        return pred

    def get_config(self):
        return {"hidden_units": hidden_dim,
                "activation": act,
                "latent_dim": latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
