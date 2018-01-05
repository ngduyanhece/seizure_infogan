import os
import sys
import time
import models as models
import keras
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
from keras.utils import to_categorical

def train(cat_dim,noise_dim,batch_size,n_batch_per_epoch,nb_epoch,dset="seizure"):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """
    general_utils.setup_logging("IG")
    # Load and rescale data
    if dset == "seizure":
        print("loading seizure data")
        X_real_train, Y_real_train, X_real_test, Y_real_test = data_utils.load_seizure()
        # pick 1000 sample for testing
        # X_real_test = X_real_test[-1000:]
        # Y_real_test = Y_real_test[-1000:]

    img_dim = X_real_train.shape[-3:]
    epoch_size = n_batch_per_epoch * batch_size

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-4, momentum=0.9, nesterov=True)

        # Load generator model
        generator_model = models.load("generator_deconv", cat_dim, noise_dim, img_dim, batch_size, dset=dset)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator", cat_dim, noise_dim, img_dim, batch_size, dset=dset)

        generator_model.compile(loss='mse', optimizer=opt_discriminator)
        # stop the discriminator to learn while in generator is learning
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model, discriminator_model, cat_dim, noise_dim)

        list_losses = ['binary_crossentropy', 'categorical_crossentropy']
        list_weights = [1, 1]
        DCGAN_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_dcgan)

        # Multiple discriminator losses
        # allow the discriminator to learn again
        discriminator_model.trainable = True
        discriminator_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_discriminator)
        # Start training
        print("Start training")
        for e in range(nb_epoch+1):
            # Initialize progbar and batch counter
            # progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()
            print("Epoch: {}".format(e+1))
            for X_real_batch,Y_real_batch in zip(data_utils.gen_batch(X_real_train, batch_size),data_utils.gen_batch(to_categorical(Y_real_train), batch_size)):

                # Create a batch to feed the discriminator model
                X_disc_fake, y_disc_fake, noise_sample = data_utils.get_disc_batch(X_real_batch,Y_real_batch, generator_model, batch_size, cat_dim, noise_dim,type="fake")
                X_disc_real, y_disc_real = data_utils.get_disc_batch(X_real_batch,Y_real_batch, generator_model, batch_size, cat_dim, noise_dim,type="real")

                # Update the discriminator
                disc_loss_fake = discriminator_model.train_on_batch(X_disc_fake, [y_disc_fake, Y_real_batch])
                disc_loss_real = discriminator_model.train_on_batch(X_disc_real, [y_disc_real, Y_real_batch])
                disc_loss = disc_loss_fake + disc_loss_real
                # Create a batch to feed the generator model
                # X_noise, y_gen = data_utils.get_gen_batch(batch_size, cat_dim, noise_dim)

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch([Y_real_batch, noise_sample], [y_disc_real, Y_real_batch])
                # Unfreeze the discriminator
                discriminator_model.trainable = True
                # training validation
                p_real_batch, p_Y_batch = discriminator_model.predict(X_real_batch, batch_size=batch_size)
                acc_train = data_utils.accuracy(p_Y_batch, Y_real_batch)
                batch_counter += 1
                # progbar.add(batch_size, values=[("D tot", disc_loss[0]),
                #                                 ("D cat", disc_loss[2]),
                #                                 ("G tot", gen_loss[0]),
                #                                 ("G cat", gen_loss[2]),
                #                                 ("P Real:", p_real_batch),
                #                                 ("Q acc", acc_train)])

                # Save images for visualization
                # if batch_counter % (n_batch_per_epoch / 2) == 0 and e % 10 == 0:
                #     data_utils.plot_generated_batch(X_real_batch, generator_model, batch_size, cat_dim, noise_dim,e)
                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
            print()
            if e % 10 == 0:
                _, p_Y_train = discriminator_model.predict(X_real_train, batch_size=X_real_train.shape[0])
                acc_train = data_utils.accuracy(p_Y_train, to_categorical(Y_real_train))
                print("Epoch: {} --- Train Accuracy: {}".format(e + 1, acc_train))
                _, p_Y_test = discriminator_model.predict(X_real_test,batch_size=X_real_test.shape[0])
                acc_test = data_utils.accuracy(p_Y_test, to_categorical(Y_real_test))
                print("Epoch: {} --- Test Accuracy: {}".format(e +1 , acc_test))
            if e % 100 == 0:
                gen_weights_path = os.path.join('../../models/IG/gen_weights.h5')
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/IG/disc_weights.h5')
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/IG/DCGAN_weights.h5')
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
