import keras
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K


def generator_upsampling(cat_dim, noise_dim, img_dim, model_name="generator_upsampling", dset="mnist"):
    """
    Generator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    s = img_dim[1]
    f = 128

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    output_channels = img_dim[-1]

    cat_input = Input(shape=cat_dim, name="cat_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    gen_input = merge([cat_input, noise_input], mode="concat")

    x = Dense(1024)(gen_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Upscaling blocks
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Conv2D(nb_filters, (4, 4), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(output_channels, (4, 4), name="gen_Conv2D_final", padding="same", activation='tanh')(x)

    generator_model = Model(inputs=[cat_input, cont_input, noise_input], outputs=[x], name=model_name)

    return generator_model


def generator_deconv(cat_dim, noise_dim, img_dim, batch_size, model_name="generator_deconv", dset="mnist"):
    """
    Generator model of the DCGAN

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    s = img_dim[1]
    f = 128

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    output_channels = img_dim[-1]

    cat_input = Input(shape=cat_dim, name="cat_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    gen_input = merge([cat_input, noise_input], mode="concat")

    x = Dense(1024)(gen_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Transposed conv blocks
    for i in range(nb_upconv - 1):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (4, 4), output_shape=o_shape, strides=(1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    # Last block
    x = UpSampling2D(size=(1, 2))(x)
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconv2D(output_channels, (4, 4), output_shape=o_shape, strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    generator_model = Model(inputs=[cat_input, noise_input], outputs=[x], name=model_name)

    return generator_model


def DCGAN_discriminator(cat_dim, img_dim, model_name="DCGAN_discriminator", dset="mnist"):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    disc_input = Input(shape=img_dim, name="discriminator_input")

    if dset == "mnist":
        list_f = [128]

    else:
        list_f = [64, 128, 256]

    # First conv
    x = Conv2D(64, (4, 4), strides=(2, 2), name="disc_Conv2D_1", padding="same")(disc_input)
    x = LeakyReLU(0.1)(x)

    # Next convs
    for i, f in enumerate(list_f):
        name = "disc_Conv2D_%s" % (i + 2)
        x = Conv2D(f, (4, 4), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)


    # More processing for auxiliary Q
    x_Q = Dense(128)(x)
    x_Q = BatchNormalization()(x_Q)
    x_Q = LeakyReLU(0.1)(x_Q)
    x_Q_Y = Dense(cat_dim[0], activation='softmax', name="Q_cat_out")(x_Q)

    # Create discriminator model
    x_disc = Dense(1, activation='sigmoid', name="disc_out")(x)
    discriminator_model = Model(inputs=[disc_input], outputs=[x_disc, x_Q_Y], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, cat_dim, noise_dim):

    cat_input = Input(shape=cat_dim, name="cat_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    generated_image = generator([cat_input, noise_input])
    x_disc, x_Q_Y = discriminator_model(generated_image)

    DCGAN = Model(inputs=[cat_input, noise_input],
                  outputs=[x_disc, x_Q_Y],
                  name="DCGAN")

    return DCGAN


def load(model_name, cat_dim, noise_dim, img_dim, batch_size, dset="mnist"):

    if model_name == "generator_upsampling":
        model = generator_upsampling(cat_dim, noise_dim, img_dim, model_name=model_name, dset=dset)
        model.summary()
        # from keras.utils import plot_model
        # plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "generator_deconv":
        model = generator_deconv(cat_dim, noise_dim, img_dim, batch_size, model_name=model_name, dset=dset)
        model.summary()
        # from keras.utils import plot_model
        # plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(cat_dim, img_dim, model_name=model_name, dset=dset)
        model.summary()
        # from keras.utils import plot_model
        # plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
