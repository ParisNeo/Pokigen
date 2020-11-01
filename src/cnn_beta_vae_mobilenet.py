"""
CNN Beta Vae Net 128
Author  : Saifeddine ALOUI
Licence : MIT
Description :
    This is a simple CNN based beta Variational auto encoder for
    128X128X3 image generation

    The variational autoencoder encodes the images in a latent space as
    a random variable with gaussian distribution then decodes them back

    The encoder goes from 128X128X3 to a list of [N,N,N] dimensions [Mean, Var, sample]
    The decoder goes from N to 128X128X3 (it takes the generated sample and generate the output image)

    First build the CNNBetaVAE128 object by passing in the
    To learn, you only need to call the learn method and pass the NX128X128X3 numpy array
    containing the images to learn from.

    To generate the images, you can sample from the latent space
"""

# OS useful to load files etc
import os

# Numpy to do array manipulation
import numpy as np

# Tensorflow/Keras layers, models, backend and callbacks
import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Input, MaxPool2D, Flatten, UpSampling2D, Reshape, Lambda, Concatenate, BatchNormalization, Add
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
tf.compat.v1.disable_eager_execution()

from cnn_beta_vae import RootCNNBetaVAE
# Open Cv for image manipulation and preproicessing
import cv2


class CNNBetaVAE(RootCNNBetaVAE):
    """
    CNN Beta Variational Encoder 128
    """
    def __init__(self, image_shape, latent_size, beta, model_weights_path = "model.h5", use_logvar=False):
        """
        Builds the model
        image_shape : The input Image Shape for example for an RGB picture of 128X128
                      you put (128,128,3)
        latent__size : The size of latent space  example 128 or even 1 for binary images generation
        beta : The weight to apply on the KL divergence (like in the beta vae paper)
        """
        # Save parameters
        RootCNNBetaVAE.__init__(self, image_shape, latent_size, beta, model_weights_path, use_logvar)

        # Recover mobilenet
        base_model = tensorflow.keras.applications.MobileNet(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=image_shape,
            include_top=False)  # Do not include the ImageNet classifier at the top.
        # Freese base model
        base_model.trainable=False
            
        # Build encoder
        input_data = Input(image_shape,name="input")
        x= base_model(input_data)
        x = Flatten()(x)
        x = Dense(4*4,activation="tanh")(x)
        # Reparametrization trick
        self.encoded_mean = Dense(latent_size)(x)
        self.encoded_sig = Dense(latent_size)(x)
        self.encoded_Data = Lambda(self.sampling, output_shape=(latent_size,), name="Encoded")([self.encoded_mean, self.encoded_sig])

        # Decoder
        latent_input=Input((latent_size))
        x = Dense(4*4,activation="tanh")(latent_input)
        x = Reshape((4,4,1))(x)
        x = self.decoderProcessingBlock(x, 128, upscaling_factor=2)
        x = self.decoderProcessingBlock(x, 64, upscaling_factor=2)
        x = self.decoderProcessingBlock(x, 32, upscaling_factor=2)
        x = self.decoderProcessingBlock(x, 16, upscaling_factor=2)
        x = self.decoderProcessingBlock(x, 3, upscaling_factor=2)
        decoded = self.decoderProcessingBlock(x, 3)

        # Build models
        self.encoder=Model(input_data,[self.encoded_mean,self.encoded_sig,self.encoded_Data], name="encoder")
        self.decoder=Model(latent_input,decoded, name="decoder")
        self.bvae=Model(input_data,self.decoder(self.encoder(input_data)[2]), name="bvae")

        # Compile the models and show some summaries
        self.bvae.compile(loss=self.loss)

        # Show the three main components the encoder, the decoder and the full VAE
        self.encoder.summary()
        self.decoder.summary()
        self.bvae.summary()
    
