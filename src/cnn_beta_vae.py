"""
CNN Beta Vae Net 128
Author  : Saifeddine ALOUI
Licence : MIT
Description :
    This is a simple CNN based beta Variational auto encoder for
    128X128X3 image generation.

    The RootCNNBetaVAE class should be inherited and encoder and decoder models
    should be defined. Use the children classes for your application

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

# Open Cv for image manipulation and preproicessing
import cv2


class RootCNNBetaVAE(Callback):
    """
    CNN Beta Variational Encoder
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
        self.reset_buffers()
        self.image_shape = image_shape
        self.latent_size = latent_size
        self.beta = beta
        self.use_logvar =use_logvar
        self.model_weights_path = model_weights_path
    
    def loss(self, y_true, y_pred):
        """
        Loss function
        Composed of a weighted sum between two losses
        reconstruction loss : to ensure tha capability of the network to reproduce the images
        kl loss : to force the latent space to have a tight distribution 
        """
        reconstruction_loss = K.mean(K.square(y_true-y_pred))
        if self.use_logvar :
            kl_loss = - 0.5 * K.mean(1 + self.encoded_sig - K.square(self.encoded_mean) - K.exp(self.encoded_sig), axis=-1)
        else:
            kl_loss = -0.5 * K.mean(1 + K.log(K.square(self.encoded_sig)) - tf.square(self.encoded_sig) - tf.square(self.encoded_mean))

        return reconstruction_loss + self.beta* kl_loss


    def encoderProcessingBlock(self, inp, nb_kernels, kernelsize=(3,3), reduction_factor=1, activation=relu, batch_normalize=False, is_residual=False):
        """
        An encoder processing block
        inp : Input to be processed by the layer
        nb_kernels : The number of kernels to apply in convolutional layer
        kernel_size : The size of the kernel to be used in convolutional layer
        reduction_factor : maxpooling factor (if 1 or less, maxpooling is not applyed)
        activation : The activation function to be applied
        batch_normalize : A boolean to tell if a batch normalization should be applied before activation
        is_residual : A boolean to tell if this is a risidual block. Warning!! If True make sure that the 
                      input and output sizes are exactly the same (no pooling and must have the same number of channels) 
        """
        x = Conv2D(nb_kernels,kernelsize, padding="same")(inp)
        if batch_normalize:
            x = BatchNormalization()(x)
        x = activation(x)
        if is_residual:
            x = Add()([inp,x])

        if reduction_factor>1:
            x = MaxPool2D((reduction_factor,reduction_factor))(x)
        return x

    def decoderProcessingBlock(self, inp, nb_kernels, kernelsize=(3,3), upscaling_factor=1, activation=relu, batch_normalize=False, is_residual=False):
        """
        A decoder processing block
        inp : Input to be processed by the layer
        nb_kernels : The number of kernels to apply in convolutional layer
        kernel_size : The size of the kernel to be used in convolutional layer
        upscaling_factor : upscaling factor (if 1 or less, upscaling is not applyed)
        activation : The activation function to be applied
        batch_normalize : A boolean to tell if a batch normalization should be applied before activation
        is_residual : A boolean to tell if this is a risidual block. Warning!! If True make sure that the 
                      input and output sizes are exactly the same (no pooling and must have the same number of channels) 
        """
        x = Conv2D(nb_kernels,kernelsize, padding="same")(inp)
        if batch_normalize:
            x = BatchNormalization()(x)
        x = activation(x)
        if is_residual:
            x = Add()([inp,x])
        if upscaling_factor>1:
            x = UpSampling2D((upscaling_factor,upscaling_factor))(x)
        return x


    # ============ Weights loading/storing methods ==================
    def load_weights(self, model_weights_file):
        """
        Loads a weights file
        (without the right weights the network is useless and need
        to be retrained)
        model_weights_file : The file containing the weights to be used
        """
        self.bvae.load_weights(model_weights_file)

    def save_weights(self, model_weights_file):
        """
        Saves weights to a weights file
        model_weights_file : The file in which to store the weights
        """
        self.bvae.load_weights(model_weights_file)

    # =========  Data manipulation methods ==============

    def load_images(self, database_path, return_image_file_names=False):
        """
        Loads the database of images and reshape them according to the needed format
        database_path.
        
        dataase_path : The path to the folder containing database of images to be loaded
        return : A numpy array of size batch, width, height, channels containing a stacking of all loaded image files
        """

        image_files = sorted([f for f in os.listdir(database_path) if f.lower().endswith(".png") or  f.lower().endswith(".jpg")])
        images=[]
        for pokemons_file in image_files:
            img = cv2.imread(os.path.join(database_path, pokemons_file), cv2.IMREAD_UNCHANGED)
            if len(img.shape)==4:
                alpha_channel = img[:, :, 3]
                _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
                color = img[:, :, :3]
                img = cv2.bitwise_not(cv2.bitwise_not(color, mask=not mask))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(np.reshape(cv2.resize(img,(self.image_shape[0],self.image_shape[1])),(1,self.image_shape[0],self.image_shape[1],self.image_shape[2])))
        images=(np.vstack(images))

        if return_image_file_names:
            return images, image_files
        else:
            return images

    def preprocess_images(self, images):
        """
        Preprocesses images to prepare them to be used by the network
        the operation assumes the images were loaded using the load_images method that prepares
        the images and resizes them to have the right format.

        The images are in the 0-255 RGB format

        this operation will normalize their values to be between -1 and 1 (useful since we use tanh as activation function
        in all layers)

        images: A 4D vector in form : Batches, width, Height, Chanels (RGB)

        returns : The normalized images numpy array
        """
        # Normalize the images to be between -1 and 1
        images=images.astype(np.float32)/255.0
        return images

    def postprocess_outputs(self, outputs):
        """
        Postprocesses the outputsto be usable as 
        regular 255 range RGB images 
        """
        outputs=outputs*255
        return outputs

    # =================== Reparameterization Trick !! It's a VAE ================
    def sampling(self, args):
        """
        Reparameterization trick by sampling for an isotropic unit Gaussian.
        To be able to back propagate through a sampling step, this trick consists
        in changing Q(z|x) to be z_mean+z_sig*N(0,1) or z_mean+exp(0.5*z_logvar)*N(0,1)
        depending on wether you use standard deviation or logvariance when computing
        the statistics in your latent space 
        # Arguments
            args : (z_mean,z_sig)
                    z_mean     : mean of Q(z|X)
                    z_sig      : standard deviation or (log of variance if self.use_logvar=True) of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_sig = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        if self.use_logvar:
            z_log_var = z_sig
            return  z_mean + K.exp(0.5 * z_log_var) * epsilon
        else:
            return  z_mean + z_sig * epsilon        


    # ========== Network Learning and predicting ==================
    def learn(self, images, epochs=500, validation_split=0.2):
        # Fit unsupervised
        print("Fitting auto encoder")
        self.bvae.fit(
            images,
            images, 
            validation_split=validation_split,  
            epochs=epochs, 
            callbacks=[
                self, 
                EarlyStopping(patience=200), 
                ModelCheckpoint(self.model_weights_path,save_best_only=True)
                ]
        )

    def predict(self, landmarks):
        return self.bvae.predict(landmarks)

    def convert(self, image):
        decoded = self.bvae.predict(image)
        return decoded

    # =========== Learning Callbacks ================================ 
    def reset_buffers(self):
        """
        Resets the statistics buffers
        """
        self.loss_buffer=[]
        self.val_loss_buffer=[]
        self.learning_encoding_buffer=[]
        self.epoch=0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch=epoch
        self.loss_buffer.append(logs["loss"])
        self.val_loss_buffer.append(logs["val_loss"])



"""
        # Build encoder
        input_data = Input(image_shape,name="input")
        x = self.encoderProcessingBlock(input_data, 64)
        x = self.encoderProcessingBlock(x, 128)
        x = Flatten()(x)
        x = Dense(16*16,activation="tanh")(x)
        # Reparametrization trick
        self.encoded_mean = Dense(latent_size)(x)
        self.encoded_sig = Dense(latent_size)(x)
        self.encoded_Data = Lambda(self.sampling, output_shape=(latent_size,), name="Encoded")([self.encoded_mean, self.encoded_sig])
        
        # Decoder
        latent_input=Input((latent_size))
        x = Dense(16*16,activation="tanh")(latent_input)
        x = Reshape((16,16,1))(x)
        x = self.decoderProcessingBlock(x,32, upscaling_factor=2)
        x_r = self.decoderProcessingBlock(x,64, upscaling_factor=2)
        x_g = self.decoderProcessingBlock(x,64, upscaling_factor=2)
        x_b = self.decoderProcessingBlock(x,64, upscaling_factor=2)

        x_r = self.decoderProcessingBlock(x_r,128, upscaling_factor=2)
        x_g = self.decoderProcessingBlock(x_g,128, upscaling_factor=2)
        x_b = self.decoderProcessingBlock(x_b,128, upscaling_factor=2)
        
        x_r = self.decoderProcessingBlock(x_r,1, upscaling_factor=1)
        x_g = self.decoderProcessingBlock(x_g,1, upscaling_factor=1)
        x_b = self.decoderProcessingBlock(x_b,1, upscaling_factor=1)
        decoded = Concatenate(axis=3)([x_r,x_g,x_b])

"""
