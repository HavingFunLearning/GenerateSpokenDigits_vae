import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
#from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from tensorflow.keras.backend import int_shape
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

class VAE:
    
    # Constructor
    def __init__(self, input_shape, latent_dim) -> None:
        self.input_shape = (input_shape[0],input_shape[1],input_shape[2])
        self.latent_dim = latent_dim
        
        self.encoder = None
        self.decoder = None
        self._build()
    
    # Kl loss
    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss
    
    # MSE loss
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss
    
    # Using the reparametrization trick i sample from the normal distro
    def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

    # This is what we need to implement
    def _build_autoencoder(self):
        model_input = Input(shape=self.input_shape, name='input autoencoder')
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")
    
    def _build_encoder(self):
        '''  This creates the encoder part:
        - the output is the sample from the posterior distro on the latent space
        '''
        encoder_input = Input(shape=self.input_shape, name='encoder_input')
        
        x = Conv2D(32, 3, padding='same', activation='relu')(encoder_input)
        x = Conv2D(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        
        self.conv_shape = int_shape(x) #Shape of conv to be provided to decoder

        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        
        # Distro Parameters
        self.z_mu = Dense(self.latent_dim, name='latent_mu')(x)   #Mean values of encoded input
        self.log_variance = Dense(self.latent_dim, name='latent_sigma')(x)  #Std dev. (variance) of encoded input
        
        z = Lambda(self.sample_point_from_normal_distribution,output_shape = K.int_shape(self.z_mu),
                   name="encoder_output")([self.z_mu, self.log_variance])
        
        self.encoder = Model(encoder_input,z, name="encoder")


    def _build_decoder(self):
        decoder_input = Input(shape=K.int_shape(self.z_mu), name="decoder_input")
        
        # This is to create a shape that can be reverted to the original size of the image
        x = Dense(self.conv_shape[1]*self.conv_shape[2]*self.conv_shape[3], activation='relu')(decoder_input) # this is a vector
        x = Reshape((self.conv_shape[1], self.conv_shape[2],self.conv_shape[3]))(x) # this is a matrix

        x = Conv2DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2))(x)
        x = Conv2DTranspose(self.input_shape[2], 3, padding='same', activation='sigmoid', name='decoder_output')(x)
        
        self.decoder = Model(decoder_input, x, name='decoder')
        
        
        
    def _build(self):
        self._build_encoder()
        self.encoder.summary()
        self._build_decoder()
        self.decoder.summary()
        self._build_autoencoder()
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.compile()
        self.model.fit(x_train, batch_size, num_epochs)


        
       
    

    