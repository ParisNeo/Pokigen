"""
Training code for Pokigen application
Licence : MIT
Author  : Saifeddine ALOUI
Description :
    Uses a database of images to build an image generator
    This code is built to use the cnnbetavae network to learn how to generate pokemons


"""
import argparse
import os

# Preprare to parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-db","--database", help="Database folder containing the images to be used to learn")
parser.add_argument("-m","--model_weights_file", help="Link to the model weights file (useful to resume learning and to put checkpoints)")
parser.add_argument("-r","--restart", action="store_true", help="if set the model will be reset, otherwise last checkpoint will be loaded")
parser.add_argument("-e","--epochs", type=int, help="The number of epochs for training")
parser.add_argument("-n","--net", help="The network to be used training")

parser.add_argument("-l","--latent_size", type=int, help="Latent space size (default to 1)")
parser.add_argument("-b","--beta", type=float, help="The value for beta (default 0.001)")

args = parser.parse_args()

# Set default arguments
if args.database is None:
    args.database = "database"

if args.epochs is None:
    args.epochs = 500

if args.model_weights_file is None:
    args.model_weights_file = "models/model_weights.h5"

if args.beta is None:
    args.beta = 2
    
if args.latent_size is None:
    args.latent_size = 27
    
if args.net is None:
    print("Using full network")
    #from cnn_beta_vae import CNNBetaVAE
    from cnn_beta_vae_mobilenet import CNNBetaVAE
    #from cnn_beta_vae_light import CNNBetaVAE
else:
    if args.net=="cnn_beta_vae":
        print("Using full network")
        from cnn_beta_vae import CNNBetaVAE
    elif args.net=="cnn_beta_vae_mobilenet":
        print("Using mobilenet network")
        from cnn_beta_vae_mobilenet import CNNBetaVAE
    else: 
        print("Using light network")
        from cnn_beta_vae_light import CNNBetaVAE
args.restart = True
# Do the training process
def main():
    """
    Main code to train the Pokigen CNN Beta VAE network
    """
    cnnbvae = CNNBetaVAE((128,128,3),args.latent_size,args.beta,model_weights_path=args.model_weights_file,use_logvar=True)

    # Check if the weights file exists and that the reset option was not issued
    if os.path.exists(args.model_weights_file) and not args.restart:
        cnnbvae.load_weights(args.model_weights_file)
    

    # Load training/validation database and preprocess it
    images = cnnbvae.preprocess_images(cnnbvae.load_images(args.database))

    # Train the network
    cnnbvae.learn(images, epochs=args.epochs)

    # Plot loss
    plt.figure()
    plt.grid()
    plt.plot(cnnbvae.loss_buffer, label="Loss")
    plt.plot(cnnbvae.val_loss_buffer, label="Validation Loss")
    plt.title("Loss curves")
    plt.savefig("../figures/{}_loss.png".format(args.net))

if __name__ == "__main__":
    # execute only if run as a script
    main()
