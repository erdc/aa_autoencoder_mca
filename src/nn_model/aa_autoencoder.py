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


class Encoder(layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, )

        self.l2 = regularizers.l2(kwargs['regu'])
        self.latent_dim = latent_dim
        self.activation = act
        self.hidden_units = hidden_dim[1:]
        self.input_dim = hidden_dim[0]
        self.input_layer  = InputLayer(input_shape=(self.input_dim,))
        self.output_layer = Dense(self.latent_dim, activation='linear')
        self.hidden_layers = [Dense(u,activation=self.activation,kernel_regularizer=self.l2) for u in self.hidden_units]

    def call(self, input_features):
        x = self.input_layer(input_features)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "l2": self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Shift(layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name ="shift", **kwargs):
        super(Shift, self).__init__(name=name, )

        self.l2 = regularizers.l2(kwargs['regu'])

        self.activation = act
        self.hidden_units = np.flip(hidden_dim)
        self.input_dim = latent_dim
        if kwargs['augment_output']:
            self.output_dim = self.hidden_units[-1]
        else:
            self.output_dim = self.hidden_units[-1]-1

        self.input_layer  = InputLayer(input_shape=(self.input_dim,))
        self.output_layer = Dense(self.output_dim, activation='linear')
        self.hidden_layers = [Dense(u,activation=self.activation,kernel_regularizer=self.l2) for u in self.hidden_units[:-1]]

    def call(self, latent_code):
        x = self.input_layer(latent_code)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "input_dim": self.input_dim,
                "output_dim":self.output_dim,
                "l2": self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(layers.Layer):

    def __init__(self, latent_dim, act, hidden_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, )

        self.l2 = regularizers.l2(kwargs['regu'])

        self.activation = act
        self.hidden_units = np.flip(hidden_dim)
        self.input_dim = latent_dim
        if kwargs['augment_output']:
            self.output_dim = self.hidden_units[-1]
        else:
            self.output_dim = self.hidden_units[-1]-1

        self.input_layer  = InputLayer(input_shape=(self.input_dim,))
        self.output_layer = Dense(self.output_dim, activation='linear')
        self.hidden_layers = [Dense(u,activation=self.activation,kernel_regularizer=self.l2) for u in self.hidden_units[:-1]]

    def call(self, latent_code):
        x = self.input_layer(latent_code)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "input_dim": self.input_dim,
                "output_dim":self.output_dim,
                "l2": self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AAautoencoder(tf.keras.Model):

    def __init__(self, latent_dim, act, hidden_dim, augment_output = True, name="autoencoder", **kwargs):
        super(AAautoencoder, self).__init__(name=name, **kwargs)

        try:
            self.l2_lam = kwargs['l2_lam']
        except:
            self.l2_lam = 1e-6


        self.augment_output = augment_output

        self.latent_dim = latent_dim
        self.activation = act
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(latent_dim, act, hidden_dim,regu=self.l2_lam)
        self.shift = Shift(latent_dim, act, hidden_dim,augment_output = self.augment_output,regu=self.l2_lam)
        self.decoder = Decoder(latent_dim, act, hidden_dim,augment_output = self.augment_output,regu=self.l2_lam)

    def call(self, input_features):
        encoded = self.encoder(input_features)
        shift = self.shift(encoded) ##Needed to build tf.graph
        pred = self.decoder(encoded)
        return pred

    def get_config(self):
        return {"hidden_units": self.hidden_dim,
                "activation": self.activation,
                "latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def save_model(u_model, u_input, mnum, results):
    ## To use TF SavedModel format with a custom training loop,
    ## call model.predict() on some input tensors first.
    ## Otherwise TF doesn't know the shape and dtype of input data
    ## it should be expecting, and thus cannot create it's weight
    ## variables. When using model.fit() this step happens automatically.

    test_predict = u_model.predict(u_input)

    u_model.save(results['savedir']+'/u_autoenc')

    np.savez_compressed(results['savedir']+'/model_history_%s'%mnum,
                        loss = results['loss'], valloss = results['valloss'],
                        shiftloss = results['shiftloss'],
                        reconloss = results['reconloss'],
                        lr = results['lr'], epochs = results['epochs'],
                        umax = results['umax'], umin = results['umin'],
                        augment = results['augment'],
                        msg=results['msg'])


def load_model(savedir,mnum):
    ## When using custom loss functions while training, there are two ways
    ## to load a saved model
    ## 1) If loaded model will not be used for retraining, then
    ##   use 'compile=False' option while loading so that TF does
    ##   search for custom objects loss functions
    u_model = tf.keras.models.load_model(savedir+'/u_autoenc', compile=False,
#                                            custom_objects={"AAautoencoder": AAautoencoder,
#                                                            "Encoder": Encoder,
#                                                            "Shift": Shift,
#                                                            "Decoder": Decoder}
                                        )
    results = np.load(savedir+'/model_history_%s.npz'%mnum)

    return u_model, results
