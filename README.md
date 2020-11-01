# Pokigen

## Description
A simple CNN Beta-VAE based Pokémon image generator written in simplified Tensorflow 2.3 Keras.
The CNN beta-VAE structure features a 
- Variational Convolution based Encoder encoding the images in a latent space as a Gaussian distribution consisting mainly in encoding blocks with different parameters depending on the version of the network you are using.
- Convolutional based decoder with upscaling decoding images from latent space back to normale images.

We present 3 varients of the network :
  - Light version : a simple CNN maxpooling, cnn upsampling architecture
  - Mobilenet : uses the convolutional layers of mobile net in encoder and cnn upsampling layers for decoding
  - Full : a fully fledged CNN encoder decoder with deep resnet structure. Needs alot of time and power to learn

The input of the autoencoder is a 128X128X3 image
The output is also a 128X128X3 image

# Objective
As for today, the software is a simple beta vae based pokemon images generator. The real objective is to correlate the pokemon name with its photo and to be able to generate pokemons from the name.


