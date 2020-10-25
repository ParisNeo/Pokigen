"""
Training code for Pokigen application
Licence : MIT
Author  : Saifeddine ALOUI
Description :
    Uses a database of images to build an image generator
    This code is built to use the cnnbetavae network to learn how to generate pokemons


"""
import argparse
from cnn_beta_vae import CNNBetaVAE
import os

# Preprare to parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-db","--database", help="Database folder containing the images to be used to learn")
parser.add_argument("-m","--model_weights_file", help="Link to the model weights file (useful to resume learning and to put checkpoints)")
parser.add_argument("-r","--restart", action="store_true", help="if set the model will be reset, otherwise last checkpoint will be loaded")
parser.add_argument("-e","--epochs", help="The number of epochs for training")
args = parser.parse_args()

# Set default arguments
if args.database is None:
    args.database = "../database"

if args.epochs is None:
    args.epochs = 500

if args.model_weights_file is None:
    args.model_weights_file = "../model/model_weights.h5"

# Do the training process
def main():
    """
    Main code to train the Pokigen CNN Beta VAE network
    """
    cnnbvae = CNNBetaVAE((128,128,3),128,10)

    # Check if the weights file exists and that the reset option was not issued
    if os.path.exists(args.model_weights_file) and not args.restart:
        cnnbvae.load_weights(args.model_weights_file)
    

    # Load training/validation database and preprocess it
    images = cnnbvae.preprocess_images(cnnbvae.load_images(args.database))

    # Train the network
    cnnbvae.learn(images, epochs=args.epochs)

    

if __name__ == "__main__":
    # execute only if run as a script
    main()