# Inhedrited from Deepmicro, defines classes for auto-encoders

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Reshape, Flatten
from keras.layers import Conv1D, Conv1DTranspose, MaxPool1D, Cropping1D, UpSampling1D
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Cropping2D, UpSampling2D
from keras import backend as K
from keras.losses import mse, binary_crossentropy


# create MLP model
def mlp_model(input_dim, numHiddenLayers=3, numUnits=64, dropout_rate=0.5):
    model = Sequential()

    # Check number of hidden layers
    if numHiddenLayers >= 1:
        # First Hidden layer
        model.add(Dense(numUnits, input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))

        # Second to the last hidden layers
        for i in range(numHiddenLayers - 1):
            numUnits = numUnits // 2
            model.add(Dense(numUnits, activation='relu'))
            model.add(Dropout(dropout_rate))

        # output layer
        model.add(Dense(1, activation='sigmoid'))

    else:
        # output layer
        model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')  # metrics=['accuracy'])
    return model


def autoencoder(dims, act='relu', init='glorot_uniform', latent_act=False, output_act=False):
    """
    Fully connected auto-encoder model, symmetric. SAE or DAE.

    Parameters
    ----------
    dims : int array
        List of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is the hidden layer.
        The decoder is symmetric with the encoder. So the total number of layers of the
        auto-encoder is 2*len(dims)-1.

    act : str
        Activation function. Not applied to Input, Hidden and Output layers.

    Returns
    ----------
    (ae_model, encoder_model) : Model of the autoencoder and model of the encoder
    """

    # whether to put an activation function in latent layer
    if latent_act:
        l_act = act
    else:
        l_act = None

    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between the input and latent layer
    n_internal_layers = len(dims) - 2

    # input layer
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers of the encoder
    for i in range(n_internal_layers):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # latent layer
    # This is the bottleneck layer from which the  features are extracted.
    h = Dense(dims[-1], activation=l_act, kernel_initializer=init, name='encoder_%d_bottle-neck' % (n_internal_layers))(h)
    y = h

    # internal layers of the decoder
    for i in range(n_internal_layers, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], activation=o_act, kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


def conv_1D_autoencoder(input_len, num_internal_layers, num_filters=3, use_max_pooling=False,
                        init='glorot_uniform', act='relu', output_act=False, rf_rate=0.1, st_rate=0.25):
    """
    Train the convolutional autoencoder (CAE) with 1D kernels

    Parameters
    ----------
    input_len : int
        The length of the feature vector. Required.

    num_internal_layers : int
        Number of dimensions to include in the intermediate layer (i.e. other than the
        input and latent layers themselves). Defaults to 1.

    num_filters : int >= 1
        Number of filters to use in each convolution layer, 3 by default.

    use_max_pooling : bool
        Whether to have max-pooling layers between internal convolution layers. By default,
        this is set to False and the only dimensionality reduction is tuned by st_rate.

    init : str
        Initialisation function. 'glorot_uniform' by default.

    act : str
        Activaton function to use. 'relu' by default.

    output_act : bool
        Whether to use activation function for the output layer. False by default.

    rf_rate : float in [0,1]
        Receptive Field (RF) size experessed as a fraction of the input layer dimension.
        0.1 by default.

    st_rate : float
        Stride size experessed as a fraction of receptive field size. 0.25 by default.
    """

    # whether to put an activation function in the output layer
    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between the input and latent layers
    if num_internal_layers < 1:
        raise ValueError("The number of internal layers for CAE should be >=1 (%d was provided)" % num_internal_layers)

    # input layer
    x = Input(shape=(input_len, 1), name='input')
    h = x

    # Keep track of the receptive field size (i.e kernel size) and stride size
    rf_size_list = []
    stride_size_list = []

    # internal layers of the encoder
    for i in range(num_internal_layers):
        rf_size = max(1, int(K.int_shape(h)[1] * rf_rate))
        rf_size_list.append(rf_size)

        stride_size = max(1, int(rf_size * st_rate))
        stride_size_list.append(stride_size)
        print("rf_size: %d, st_size: %d" % (rf_size, stride_size))

        h = Conv1D(filters=num_filters, kernel_size=rf_size, strides=stride_size, activation=act, padding='same', kernel_initializer=init, name='encoder_conv_%d' % i)(h)
        if use_max_pooling:
            h = MaxPool1D(pool_size=2, padding='same')(h)

    final_shape = K.int_shape(h)[1:]

    # bottleneck layer, features are extracted from here
    h = Flatten(name='flatten')(h)
    y = h
    y = Reshape(final_shape)(y)

    # internal layers of the decoder
    for i in range(num_internal_layers-1, -1, -1):
        if use_max_pooling:
            y = UpSampling1D(2)(y)

        if i == 0:
            # Output layer, some params might be set differently
            y = Conv1DTranspose(filters=1, kernel_size=rf_size_list[0], strides=stride_size_list[0], activation=o_act, padding='same', kernel_initializer=init, name='decoder_conv_0')(y)
        else:
            # Internal layer
            y = Conv1DTranspose(filters=num_filters, kernel_size=rf_size_list[i], strides=stride_size_list[i], activation=act, padding='same', kernel_initializer=init, name='decoder_conv_%d' % i)(y)

    # Output cropping
    # If the stride_size is not a divisor of the input length, there might be extra dimensions
    cropping_size = K.int_shape(y)[1] - K.int_shape(x)[1]
    if cropping_size > 0:
        y = Cropping1D(cropping=(0, cropping_size), name='crop')(y)

    return Model(inputs=x, outputs=y, name='CAE'), Model(inputs=x, outputs=h, name='encoder')


def conv_2D_autoencoder(input_len, num_internal_layers, num_filters=3, use_max_pooling=False,
                        init='glorot_uniform', act='relu', output_act=False, rf_rate=0.1, st_rate=0.25):
    """
    Train the convolutional autoencoder (CAE) with 2D kernels

    The feature vector is pre-converted into a square matrix before feeding and this
    model scans it as if it is an "image"

    Parameters
    ----------
    input_len : int
        The dimension of the square feature matrix. Required.

    num_internal_layers : int
        Number of dimensions to include in the intermediate layer (i.e. other than the
        input and latent layers themselves). Defaults to 1.

    num_filters : int >= 1
        Number of filters to use in each convolution layer, 3 by default.

    use_max_pooling : bool
        Whether to have max-pooling layers between internal convolution layers. By default,
        this is set to False and the only dimensionality reduction is tuned by st_rate.

    init : str
        Initialisation function. 'glorot_uniform' by default.

    act : str
        Activaton function to use. 'relu' by default.

    output_act : bool
        Whether to use activation function for the output layer. False by default.

    rf_rate : float in [0,1]
        Receptive Field (RF) size experessed as a fraction of the input layer dimension.
        0.1 by default.

    st_rate : float
        Stride size experessed as a fraction of receptive field size. 0.25 by default.
    """

    # whether to apply activation function in the output layer
    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between the input and latent layers
    if num_internal_layers < 1:
        raise ValueError("The number of internal layers for CAE should be >=1 (%d was provided)" % num_internal_layers)

    # input layer
    x = Input(shape=(input_len, input_len, 1), name='input')
    h = x

    # Keep track of the receptive field size (i.e kernel size) and stride size
    rf_size_list = []
    stride_size_list = []

    # internal layers in encoder
    for i in range(num_internal_layers):
        rf_size = max(1, int(K.int_shape(h)[1] * rf_rate))
        rf_size_list.append(rf_size)

        stride_size = max(1, int(rf_size * st_rate))
        stride_size_list.append(stride_size)
        print("rf_size: %d, st_size: %d" % (rf_size, stride_size))

        h = Conv2D(filters=num_filters, kernel_size=rf_size, strides=stride_size, activation=act, padding='same', kernel_initializer=init, name='encoder_conv_%d' % i)(h)
        if use_max_pooling:
            h = MaxPool2D(pool_size=(2, 2), padding='same')(h)

    reshapeDim = K.int_shape(h)[1:]

    # bottle neck layer, features are extracted from here
    h = Flatten()(h)
    y = h
    y = Reshape(reshapeDim)(y)

    # internal layers of the decoder
    for i in range(num_internal_layers-1, -1, -1):
        if use_max_pooling:
            y = UpSampling2D((2, 2))(y)

        if i == 0:
            # Output layer, some params might be set differently
            y = Conv2DTranspose(filters=1, kernel_size=rf_size_list[i], strides=stride_size_list[i], activation=o_act, padding='same', kernel_initializer=init, name='decoder_conv_0')(y)
        else:
            # Internal layer
            y = Conv2DTranspose(filters=num_filters, kernel_size=rf_size_list[i], strides=stride_size_list[i], activation=act, padding='same', kernel_initializer=init, name='decoder_conv_%d' % i)(y)

    # Output cropping
    # If the stride_size is not a divisor of the input length, there might be extra dimensions
    cropping_size = K.int_shape(y)[1] - K.int_shape(x)[1]
    if cropping_size > 0:
        y = Cropping2D(cropping=((0, cropping_size), (0, cropping_size)), name='crop')(y)

    return Model(inputs=x, outputs=y, name='CAE'), Model(inputs=x, outputs=h, name='encoder')


# Variational Autoencoder
def variational_AE(dims, act='relu', init='glorot_uniform', output_act=False, recon_loss='mse', beta=1):
    """
    Train the variational autoencoder (VAE)

    Parameters
    ----------
    dims : int or ndarray of shape (`n_layers`)
        Number of dimensions to include in the latent layer (and other intermediate layers)

    act : str
        Actovation function to use. 'relu' by default.

    output_act : bool
        Whether to use an activation function for the output layer. False by default.

    recon_loss : str
        Method to evaluate reconstruction loss by the VAE. 'mse' by default.

    beta : float
        Weight for the Kullback-Leibler divergence. Defaults to 1.
    """

    if output_act:
        o_act = 'sigmoid'
    else:
        o_act = None

    # The number of internal layers: layers between input and latent layer
    n_internal_layers = len(dims) - 2

    # build encoder model
    inputs = Input(shape=(dims[0],), name='input')
    h = inputs

    # internal layers in encoder
    for i in range(n_internal_layers):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # latent layer
    z_mean = Dense(dims[-1], name='z_mean')(h)
    z_sigma = Dense(dims[-1], name='z_sigma')(h)

    # Use reparameterization trick to push the sampling out as input
    # See https://www.tensorflow.org/tutorials/generative/cvae
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.

        Parameters
        ----------
            args : tensor
            Mean and log of variance of Q(z|X)

        Returns
        ----------
            z : tensor
            Sampled latent vector
        """

        z_mean, z_sigma = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + z_sigma * epsilon

    z = Lambda(sampling, output_shape=(dims[-1],), name='z')([z_mean, z_sigma])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_sigma, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(dims[-1],), name='z_sampling')
    y = latent_inputs

    # internal layers in decoder
    for i in range(n_internal_layers, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    outputs = Dense(dims[0], kernel_initializer=init, activation=o_act)(y)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # loss function
    if recon_loss == 'mse':
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs, outputs)

    reconstruction_loss *= dims[0]

    kl_loss = 1 + K.log(1e-8 + K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + (beta * kl_loss))
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam')

    vae.metrics.append(K.mean(reconstruction_loss))
    vae.metrics_names.append("recon_loss")
    vae.metrics.append(K.mean(beta * kl_loss))
    vae.metrics_names.append("kl_loss")

    return vae, encoder, decoder
