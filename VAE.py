#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Layer, Lambda
#from keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        ''' the collection of losses will be aggregated and added to the specified Keras loss function 
        to form the loss we ultimately minimize
        '''
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
########################################################################################

class VAE:
    
    # Constructor
    def __init__(self, input_shape, latent_dim) -> None:
        self.input_shape = (input_shape[0],input_shape[1],input_shape[2])
        self.latent_dim = latent_dim
        self.reconstruction_loss_weight = 10000
        
        self.encoder = None
        self.decoder = None
        self._build()
    

    # MSE loss
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

   ########################################################################################
    
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
        mu = Dense(self.latent_dim, name='latent_mu')(x)   #Mean values of encoded input
        log_variance = Dense(self.latent_dim, name='latent_sigma')(x)  #Std dev. (variance) of encoded input
        
        mu, log_variance = KLDivergenceLayer()([mu, log_variance])

        
        def sample_point_from_normal_distribution(args):
                mu, log_variance = args
                batch = K.shape(mu)[0]
                dim = K.int_shape(mu)[1]
                epsilon = K.random_normal(shape=(batch, dim), mean=0., 
                                        stddev=1.)
                sampled_point = mu + K.exp(log_variance / 2) * epsilon
                return sampled_point
        
        z = Lambda(sample_point_from_normal_distribution,output_shape=(self.latent_dim, ),
                   name="encoder_output")([mu, log_variance])
        
        self.encoder = Model(encoder_input,z, name="encoder")
        
   ########################################################################################


    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim, ), name="decoder_input")
        
        # This is to create a shape that can be reverted to the original size of the image
        x = Dense(self.conv_shape[1]*self.conv_shape[2]*self.conv_shape[3], activation='relu')(decoder_input) # this is a vector
        x = Reshape((self.conv_shape[1], self.conv_shape[2],self.conv_shape[3]))(x) # this is a matrix
        
        x = Conv2DTranspose(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
        x = Conv2DTranspose(32, 3, padding='same', activation='relu')(x)

        x = Conv2DTranspose(1, 3, padding='same', activation='sigmoid', name='decoder_output')(x)
        dec_out = x
        self.decoder = Model(decoder_input, dec_out, name='decoder')
        
        
        
    def _build(self):
        self._build_encoder()
        self.encoder.summary()
        self._build_decoder()
        self.decoder.summary()
        self._build_autoencoder()
        
        
        
        
        
    
    
     # This is what we need to implement
    def _build_autoencoder(self):
        model_input = Input(shape=self.input_shape, name='input_autoencoder')
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")
    
    
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_reconstruction_loss,
                           metrics=[],
                                    )
    
    def train(self, x_train, x_test, batch_size, num_epochs):
        # Check input shape
        print("Input shape of x_train:", x_train.shape)
        # Compile the model
        self.compile()
        # Train the model
        print("Training the model...")
        history = self.model.fit(x_train, x_train, validation_data = (x_test,x_test),epochs=num_epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")


        
       
    

        