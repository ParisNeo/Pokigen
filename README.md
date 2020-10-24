# Pokigen

A simple CNN Beta-VAE based Pokémon image generator written in simplified Tensorflow 2.3 Keras.
The CNN beta-VAE structure features a 
- 5 Convolutional/Maxpooling layers encoder followed by a Dense layer containing means and variances of the latent space (the size is adjustable by user)
- 5 Deconvolutional/UpSampling layers decoder starting with a dense layer

The input of the autoencoder is a 128X128X3 image
The output is also a 128X128X3 image

The project is still in progress.

The final objective is to be able to generate a pokemon image from the pokemon name and a random seed input.

