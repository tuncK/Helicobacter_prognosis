# Detailed definition of LSTM/attention based autoencoders

from keras.models import Model
from keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed


def ae(dims, act='relu', init='glorot_uniform', latent_act=False, output_act=False):
    """
    LSTM-based auto-encoder model

    Performs a time-series like analysis, but without window-based pre-processing.

    Parameters
    ----------
    dims : int array
        List of number of features in each layer of the encoder. dims[0] is input dim,
        dims[-1] is the size of the latent layer. The decoder is symmetric with the encoder.
        So the total number of LSTM units of the auto-encoder is 2*len(dims)-2.

    act : str
        Activation function to use. 'relu' by default.

    latent_act : bool
        Whether to use activation function for the latent layer. False by default.

    output_act : bool
        Whether to use activation function for the output layer. False by default.

    Returns
    ----------
    (ae_model, encoder_model) : Model of the autoencoder and model of the encoder
    """

    # whether to put an activation function in the latent layer
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
    x = Input(shape=(dims[0], 1), name='input')
    h = x

    # internal layers of the encoder
    for i in range(n_internal_layers):
        h = LSTM(dims[i + 1], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_%d' % i)(h)

    # latent layer
    # This is the bottleneck layer from which the  features are extracted.
    h = LSTM(dims[-1], activation=l_act, kernel_initializer=init, return_sequences=False, name='encoder_%d_bottleneck' % n_internal_layers)(h)
    y = h

    y = RepeatVector(dims[0])(y)

    # internal layers of the decoder
    for i in range(n_internal_layers, 0, -1):
        y = LSTM(dims[i + 1], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_%d' % i)(y)

    # Last LSTM layer of the decoder
    y = LSTM(dims[1], activation=o_act, kernel_initializer=init, return_sequences=True, name='decoder_0')(y)

    # Output layer
    y = TimeDistributed(Dense(1), name='output')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')
