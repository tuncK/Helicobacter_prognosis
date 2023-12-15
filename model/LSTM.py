# Detailed definition of LSTM/attention based autoencoders

from keras.models import Model
from keras.layers import Dense, Input, LSTM, RepeatVector, Reshape, TimeDistributed
import numpy as np


def clip_windows(X, Y, window_size, stride_size=1):
    """
    Function to chop down over-long feature lists into predifined chunks
    so that the GPU memory does not run out.

    Parameters
    ----------
    X : array
        NxM dim array of features

    Y : array
        Nx1 dim array of labels

    window_size : int
        Max number of dimensions in the output features. If there are more, a second data entry
        will be generated with the same label.

    stride_ratio : float
        By how much the feature selection window should be slided between each feature cropping.
        Expressed as a fraction of the window_size. 0.5 by default.

    Returns
    ----------
    (xx, yy) : pair of arrays
        Kxwindow_size and Kx1 arrays after feature clipping by sliding window. K is the new (increased)
        dimensionality if feature dimensionality of the input X (M) is higher than window_size.
        K = N * ceil( M / window_size * stride_ratio )
    """

    if X.shape[0] != Y.shape[0]:
        raise ValueError('Number of samples in the features (%s) and labels (%s) do not match.' % (X.shape, Y.shape))
    elif X.shape[1] <= window_size:
        raise ValueError('Number of features in the input (%s) is fewer than the requested window length (%d).' % (X.shape, window_size))
    else:
        # Input is indeed big and needs to be chopped into smaller pieces.
        featureIDs = np.tile([x for x in range(X.shape[1])], (X.shape[0],1))
        X_ID_tensor = np.dstack((X, featureIDs))
        
        xx = np.zeros((0, window_size, 2))
        yy = np.zeros(0)
        for start_pos in range(0, X.shape[1] - window_size + 1, stride_size):
            end_pos = start_pos + window_size
            xx = np.append(xx, X_ID_tensor[:, start_pos:end_pos, :], axis=0)
            yy = np.append(yy, Y, axis=0)
        return (xx, yy)


# CuDNN kernel is only available for certain conditions:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# As a result, some model params can be adjusted differently, but at a very high
# additional training time cost.
def ae(dims, act='tanh', init='glorot_uniform'):
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
        Activation function to use. 'tanh' by default. Setting to any alternative
        will cause a significant increase in training time due to lack of CuDNN support.

    Returns
    ----------
    (ae_model, encoder_model) : Model of the autoencoder and model of the encoder
    """

    # The number of internal layers: layers between the input and latent layer
    n_internal_layers = len(dims) - 2

    # input layer
    x = Input(shape=(dims[0], 2), name='input')
    h = x

    # internal layers of the encoder
    for i in range(n_internal_layers):
        h = LSTM(dims[i + 1], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_%d' % i)(h)

    # latent layer
    # This is the bottleneck layer from which the  features are extracted.
    h = LSTM(dims[-1], activation=act, kernel_initializer=init, return_sequences=False, name='encoder_%d_bottleneck' % n_internal_layers)(h)
    y = h

    y = RepeatVector(dims[0])(y)

    # internal layers of the decoder
    for i in range(n_internal_layers, 0, -1):
        y = LSTM(dims[i + 1], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_%d' % i)(y)

    # Last LSTM layer of the decoder
    y = LSTM(dims[1], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_0')(y)

    # Output layer
    y = TimeDistributed(Dense(2), name='output')(y)

    return Model(inputs=x, outputs=y, name='LSTM_AE'), Model(inputs=x, outputs=h, name='encoder')
