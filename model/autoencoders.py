#!/usr/bin/env python

# improved version of DeepMicro
# https://www.nature.com/articles/s41598-020-63159-5
# https://github.com/minoh0201/DeepMicro


# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Bayesian optimisation for 5-fold cross validation
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# importing keras
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model
from keras.optimizers import Adam

# importing util libraries
import os
import time

# importing custom library
import DNN_models
import LSTM


# Fix a np.random.seed for reproducibility in numpy processing
np.random.seed(42)
matplotlib.use('agg')


class Modality(object):
    """
    Auto-encoder object for a stand-alone data modality

    Parameters
    ----------
    data : str
        Filename containing the entire training dataset
        File should contain an ndarray of shape (`n_features`,`n_samples`)

    clipnorm_lim : float
        Threshold for gradient normalisation for numerical stability during training.
        If the norm of the gradient exceeds this threshold, it will be scaled down to this value.
        The lower is the more stable, at the expense of increased the training duration.

    max_training_duration : int
        Maximum duration to be allowed for the AE training step to take. Walltime measured in seconds,
        by default the training duration is unlimited.

    seed : int
        Seed for the number random generator. Defaults to 0.
    """

    def __init__(self, Xfile, Yfile, clipnorm_lim=1, seed=0, max_training_duration=np.inf, **kwargs):
        self.t_start = time.time()
        self.Xfilename = str(Xfile)
        self.Yfilename = str(Yfile)
        self.data = self.Xfilename.split('/')[-1].split('_')[0].split('.')[0]
        self.seed = seed
        self.max_training_duration = max_training_duration
        self.prefix = ''
        self.representation_only = False
        self.clipnorm_lim = clipnorm_lim
        self.dataset_ids = None

        params = self.data + '_' + str(self.seed) + '_' + str(clipnorm_lim)
        self.modelName = '%s' % params

        self.output_dir = '../results/'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def load_X_data(self, dtype=None):
        """
            Read training data (X) from file
        """

        # read file
        if os.path.isfile(self.Xfilename):
            raw = pd.read_csv(self.Xfilename, sep='\t', index_col=0, header=0)
        else:
            raise FileNotFoundError("File {} does not exist".format(self.Xfilename))

        # Keep the patient ids etc. to be able to match to labels later on.
        # We will remove pandas auto-added suffixes on duplicates
        # ABC, ABC.1, ABC.2 ... -> ABC
        self.dataset_ids = [x.split('.')[0] for x in list(raw)]

        # load data
        self.X_train = raw.transpose().values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test, at least temporarily
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1, self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)

        self.printDataShapes(train_only=True)

    def load_Y_data(self, dtype=None):
        """
        Reads class labels (Y) from file
        """

        if os.path.isfile(self.Yfilename):
            imported_labels = pd.read_csv(self.Yfilename, sep='\t', index_col=0, header=0)

            # There might be duplicate measurements for the same patient.
            # i.e., some patient identifiers might need to be repeated.
            # The order of params also needs to match to the training data X
            labels = imported_labels.loc[self.dataset_ids]
        else:
            raise FileNotFoundError("{} does not exist".format(self.Yfilename))

        # Label data validity check
        if not labels.values.shape[1] > 1:
            label_flatten = labels.values.reshape((labels.values.shape[0])).astype(dtype)
        else:
            raise IndexError('The label file contains more than 1 column.')

        # train and test split
        split_data = train_test_split(self.X_train, label_flatten, test_size=0.2, random_state=self.seed, stratify=label_flatten)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        self.printDataShapes()

    def export_transformed_data(self, filename=None):
        """
            Export a dataframe with the latent representation of the training data (X) to file

            WARNING: Only exports training data, so if training gets split between
            the end of AE training and invocation of this function, the export
            will be partial.

            Parameters
            ----------
            filename : str
                Filename for the output file. If none provided, it will suffix the original
                input file with '_latent'.
        """

        if not filename:
            # By default, the output filename will be:
            # a/b.ext -> a/b_latent.ext
            filename = self.Xfilename().split('.')[0] + '_latent.tsv'

        # Get the matrix with labels
        transformed_df = pd.DataFrame(self.X_train).transpose()
        transformed_df.columns = self.dataset_ids

        # writeout
        transformed_df.to_csv(filename, sep='\t', header=True, index=True)
        print("The learned representation of the training set has been saved in '{}'".format(filename))

    # Principal Component Analysis
    def pca(self, ratio_threshold=0.99, save_model=False):
        # Generate an experiment identifier string for the output files
        self.prefix = self.prefix + 'PCA_'

        # Perform PCA
        pca = PCA()
        pca.fit(self.X_train)

        # Evaluate the number of components that collectively explain
        # threshold-fraction of the observed variance.
        ratio_sum = np.cumsum(pca.explained_variance_ratio_)
        n_comp = sum(ratio_sum < ratio_threshold) + 1

        # Repeat PCA with n-many components only.
        pca = PCA(n_components=n_comp)
        pca.fit(self.X_train)

        # applying the eigenvectors to the whole training and the test set.
        if save_model:
            self.X_train = pca.transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
            self.printDataShapes()

    # Gausian Random Projection
    def grp(self, eps=0.1, save_model=False):
        # Generate an experiment identifier string for the output files
        self.prefix = self.prefix + 'GRP_'

        # Perform GRP
        rf = GaussianRandomProjection(eps)
        rf.fit(self.X_train)

        if save_model:
            # applying GRP to the whole training and the test set.
            self.X_train = rf.transform(self.X_train)
            self.X_test = rf.transform(self.X_test)
            self.printDataShapes()

    # Custom keras callback function to limit total training time
    # This is needed for early stopping the procedure during BOHB
    class TimeLimit_Callback(keras.callbacks.Callback):
        def __init__(self, verbose=False, max_training_duration=np.inf):
            self.training_start_time = time.time()
            self.verbose = verbose
            self.max_training_duration = max_training_duration

        def on_epoch_end(self, epoch, logs={}):
            duration = time.time() - self.training_start_time
            if self.verbose:
                print('%ds passed so far' % duration)

            if duration >= self.max_training_duration:
                print('Training exceeded time limit (max=%ds), stopping...'
                      % self.max_training_duration)
                self.model.stop_training = True
                self.stopped_epoch = epoch

    # Shallow Autoencoder & Deep Autoencoder
    def sae(self, dims=[50], epochs=10000, batch_size=100, verbose=2, loss='mean_squared_error', latent_act=False,
            output_act=False, act='relu', patience=20, val_rate=0.2, save_model=False, **kwargs):

        """
        Train the shallow (1-layer) or deep (>1 layers) auto-encoder

        Parameters
        ----------
        dims : int or ndarray of shape (`n_layers`)
            Number of dimensions to include in the latent layer (and other intermediate layers)

        epochs : int
            Maximum number of epochs to train the AE. Defaults to 10000. The early stopping
            might make it stop earlier than specified here.

        verbose : int
            Verbosity level
            0 -
            1 -
            2 -

        loss : str
            Parameter to watch the loss by, dafaults to mean_squared_error ('mse').
            Refer to keras docs: https://keras.io/api/losses/

        act : str
            Activation function to use. 'relu' by default.

        latent_act : bool
            Whether to use activation function for the latent layer. False by default.

        output_act : bool
            Whether to use activation function for the output layer. False by default.

        patience : int
            Number of epochs to continue training after a non-decreasing valdation loss. 20 by default.

        val_rate : float
            Fraction of data to be set aside as the validation set. 0.2 by default.

        save_model : bool
            Whether to save the model parameters during training, false by default. Enabling during
            hyperparameter tuning might slow down the operations. Enabling during parallel execution
            might also result in a race condition if target file names are not distinct.
        """

        # Generate an experiment identifier string for the output files
        if patience != 20:
            self.prefix += 'p' + str(patience) + '_'
        if len(dims) == 1:
            self.prefix += 'AE'
        else:
            self.prefix += 'DAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if latent_act:
            self.prefix += 't'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix = self.prefix + 's'

        # callbacks for each epoch
        callbacks = self.set_callbacks(patience=patience, save_model=save_model)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train, test_size=val_rate, random_state=self.seed, stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.ae, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)

        # Train the AE
        self.train_ae(batch_size, callbacks, epochs, loss, save_model, val_rate, verbose)

    def vae(self, dims=[8], epochs=10000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu',
            patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, save_model=False, **kwargs):
        """
        Train the variational autoencoder (VAE)

        Parameters
        ----------
        dims : int or ndarray of shape (`n_layers`)
            Number of dimensions to include in the latent layer (and other intermediate layers)
            Defaults to 8.

        epochs : int
            Maximum number of epochs to train the AE. Defaults to 10000. The early stopping
            might make it stop earlier than specified here.

        verbose : int
            Verbosity level
            0 -
            1 -
            2 -

        loss : str
            Parameter to watch the loss by, dafaults to mean_squared_error ('mse').
            Refer to keras docs: https://keras.io/api/losses/

        act : str
            Actovation function to use. 'relu' by default.

        output_act : bool
            Whether to use activation function for the output layer. False by default.

        patience : int
            Number of epochs to continue training after a non-decreasing valdation loss. 25 by default.

        val_rate : float
            Fraction of data to be set aside as the validation set. 0.2 by default.

        warmup : bool
            Whether to use warmup strategy for the VAE initialisation. True by default.
            Enabling results in more active units at the early epochs, which would be gradually
            pruned away and might improve learning performance.

        warmup_rate : float

        beta : float

        save_model : bool
            Whether to save the model parameters during training, false by default. Enabling during
            hyperparameter tuning might slow down the operations. Enabling during parallel execution
            might also result in a race condition if target file names are not distinct.
        """
        save_model=True
        # Generate an experiment identifier string for the output files
        if patience != 25:
            self.prefix += 'p' + str(patience) + '_'
        if warmup:
            self.prefix += 'w' + str(warmup_rate) + '_'
        self.prefix += 'VAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        if beta != 1:
            self.prefix += 'B' + str(beta)
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # callbacks for each epoch
        callbacks = self.set_callbacks(patience=patience, save_model=save_model)

        # Add warm-up callback, if so requested
        if warmup:
            # warm-up implementation
            def warm_up(epoch):
                val = epoch * warmup_rate
                if val <= 1.0:
                    K.set_value(beta, val)

            beta = K.variable(value=0.0)
            warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])
            callbacks.append(warm_up_cb)

        # insert input shape into dimension list
        dims.insert(0, self.X_train.shape[1])

        # create vae model
        self.ae, self.encoder, self.decoder = DNN_models.variational_AE(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)

        # Train the AE
        self.train_ae(batch_size, callbacks, epochs, loss, save_model, val_rate, verbose)

    # Convolutional Autoencoder
    def cae(self, num_internal_layers=1, num_filters=3, use_2D=False, epochs=10000, batch_size=100, verbose=2,
            loss='mse', output_act=False, act='relu', patience=25, val_rate=0.2, rf_rate=0.1, st_rate=0.25,
            save_model=False, **kwargs):
        """
        Train the convolutional autoencoder (CAE)

        Parameters
        ----------
        num_internal_layers : int
            Number of dimensions to include in the intermediate layer (i.e. other than the
            input and latent layers themselves). Defaults to 1.

        use_2D : bool
            By default, 1D kernels will be used to search for patterns in the input feature
            vectors (i.e. False). Enabling will convert these feature vectors into square
            matrices with 0-padding as needed and train 2D kernels, instead.

        epochs : int
            Maximum number of epochs to train the AE. Defaults to 10000. The early stopping
            might make it stop earlier than specified here.

        verbose : int
            Verbosity level: 0 (the most quiet), 1 (intermediate) or 2 (the most detailed).

        loss : str
            Parameter to watch the loss by, dafaults to mean_squared_error ('mse').
            Refer to keras docs: https://keras.io/api/losses/

        act : str
            Actovaton function to use. 'relu' by default.

        output_act : bool
            Whether to use activation function for the output layer. False by default.

        patience : int
            Number of epochs to continue training after a non-decreasing valdation loss. 25 by default.

        val_rate : float in [0,1]
            Fraction of data to be set aside as the validation set. 0.2 by default.

        rf_rate : float in [0,1]
            Receptive Field (RF) size experessed as a fraction of the input layer dimension.
            0.1 by default.

        st_rate : float
            Stride size experessed as a fraction of receptive field size. 0.25 by default.

        save_model : bool
            Whether to save the model parameters during training, false by default. Enabling during
            hyperparameter tuning might slow down the operations. Enabling during parallel execution
            might also result in a race condition if target file names are not distinct.
        """

        # Generate an experiment identifier string for the output files
        self.prefix += 'CAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(num_internal_layers).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # callbacks for each epoch
        callbacks = self.set_callbacks(patience=patience, save_model=save_model)

        # create cae model
        if use_2D:
            # Cast the 1D feature vectors into a 2D square matrix
            # Pad with 0s as needed
            def vec2square(x):
                dim = int(np.sqrt(x.shape[1])) + 1
                padding_len = dim ** 2 - x.shape[1]
                padded = np.pad(x, ((0, 0), (0, padding_len)), 'constant', constant_values=0)
                return padded.reshape((x.shape[0], dim, dim))

            self.X_train = vec2square(self.X_train)
            self.X_test = vec2square(self.X_test)

            self.ae, self.encoder = DNN_models.conv_2D_autoencoder(input_len=self.X_train.shape[1], num_internal_layers=num_internal_layers,
                                                                   act=act, output_act=output_act, rf_rate=rf_rate, st_rate=st_rate)
        else:
            # If using a 1D CAE, no such conversion is needed. Data enters as a vector as is.
            self.ae, self.encoder = DNN_models.conv_1D_autoencoder(input_len=self.X_train.shape[1], num_internal_layers=num_internal_layers,
                                                                   act=act, output_act=output_act, rf_rate=rf_rate, st_rate=st_rate)

        # Train the AE
        self.train_ae(batch_size, callbacks, epochs, loss, save_model, val_rate, verbose)

    def lstm(self, dims=[50], epochs=10000, batch_size=100, verbose=2, loss='mean_squared_error', latent_act=False,
             output_act=False, act='relu', patience=20, val_rate=0.2, save_model=False, **kwargs):

        # Generate an experiment identifier string for the output files
        self.prefix += 'LSTM'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # callbacks for each epoch
        callbacks = self.set_callbacks(patience=patience, save_model=save_model)

        # insert input shape into dimension list
        dims.insert(0, self.X_train.shape[1])

        # Build the model
        self.ae, self.encoder = LSTM.ae(dims=dims, act=act, latent_act=latent_act, output_act=output_act)

        # Train the AE
        self.train_ae(batch_size, callbacks, epochs, loss, save_model, val_rate, verbose)

    def set_callbacks(self, patience, save_model):
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=0),
                     self.TimeLimit_Callback(max_training_duration=self.max_training_duration)]

        # Exports the model to file at each iteration.
        # Due to early stopping, the final model is not necessarily the best model.
        # Constant disk IO may slow down the training considerably.
        if save_model:
            self.model_out_file = self.output_dir + '/' + self.modelName + '.h5'
            model_write_callback = ModelCheckpoint(self.model_out_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.append(model_write_callback)

            # clean up model checkpoint before use
            if os.path.isfile(self.model_out_file):
                os.remove(self.model_out_file)

        return callbacks

    def train_ae(self, batch_size, callbacks, epochs, loss, save_model, val_rate, verbose):
        self.ae.summary()

        # compile
        customised_adam = Adam(clipnorm=self.clipnorm_lim)
        self.ae.compile(optimizer=customised_adam, loss=loss)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # fit
        self.history = self.ae.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, X_inner_test, None))

        if save_model:
            # save loss progress
            self.saveLossProgress()

            # load best model
            self.ae.load_weights(self.model_out_file)

            # Determine which layer within the model is the latent layer
            for layer in self.ae.layers:
                if 'bottleneck' in layer.name:
                    latent_layer_idx = self.ae.layers.index(layer)

            self.encoder = Model(self.ae.layers[0].input, self.ae.layers[latent_layer_idx].output)

            # applying the learned encoder into the whole training and the test set.
            self.X_train = self.encoder.predict(self.X_train)
            self.X_test = self.encoder.predict(self.X_test)
            self.printDataShapes()

    def printDataShapes(self, train_only=False):
        print("X_train.shape: ", self.X_train.shape)
        if not train_only:
            print("y_train.shape: ", self.y_train.shape)
            print("X_test.shape: ", self.X_test.shape)
            print("y_test.shape: ", self.y_test.shape)

    # ploting loss progress over epochs
    def saveLossProgress(self):
        loss_collector, loss_max_atTheEnd = self.saveLossProgress_ylim()

        # save loss progress - train and val loss only
        plt.rcParams.update({'font.size': 12})
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Autoencoder model loss')
        plt.ylabel('loss')
        plt.yscale('log')
        # plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
        plt.xlabel('Epoch')
        plt.legend(['train', 'val.'], loc='upper right')
        plt.savefig(self.output_dir + '/' + self.modelName + '.png')
        plt.close()

    # supporting loss plot
    def saveLossProgress_ylim(self):
        loss_collector = []
        loss_max_atTheEnd = 0.0
        for hist in self.history.history:
            current = self.history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_atTheEnd:
                loss_max_atTheEnd = current[-1]
        return loss_collector, loss_max_atTheEnd

    # Might need to revisit
    def get_min_loss(self):
        return min(self.history.history['val_loss'])

    # Classification
    def classification(self, method='svm', cv=5, scoring='roc_auc', n_jobs=-1, cache_size=10000, use_bayes_opt=False, subpart_dims=None, verbose=0):
        """
        Train the classifier.

        For high dimensional data, should be run after the autoencoder is trained and
        the dimensionality is reduced.

        Parameters
        ----------
        method : str, 'svm', 'rf', 'mlp' or 'mkl'
            Classifier type to be trained. SVM by default.

        cv : int
            Number of folds for the cross-validation. Defaults to 5.

        scoring : str
            Objective function to be used to assess training quality. AUC of ROC curve by default.

        n_jobs : int
            Number of parallel jobs to start, uses all available cores by default (-1).

        use_bayes_opt : bool
            Whether to use Bayesian optimisation rather than grid search. False by default.

        subpart_dims : int array
            Iff using MKL, the dimensions of each modality that contributes to the final feature vector.
            The input feature vectors will be segmented back to constitutents by these dimensions.
        """

        clf_start_time = time.time()
        print("# Tuning hyper-parameters")

        # Support Vector Machine
        if method == 'svm':
            hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                                {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)], 'kernel': ['rbf']}]
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=verbose)

        # SVM, but with multiple kernels linearly combined
        if method == 'mkl':
            from multi_kernel_SVM import multi_kernel_SVM
            mkl = multi_kernel_SVM(probability=True, cache_size=cache_size, dims=subpart_dims)

            if use_bayes_opt:
                # BO to determine next parameters to test within the pre-defined search space.
                # https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

                hyper_parameters = [{
                                    'kernel_type': Categorical(['linear', 'rbf']),
                                    }]

                # Add only as many random weight dimensions as needed here
                # so that BO search space is kept small.
                # w3 = Real(0.0,1.0, prior='uniform') etc.
                for i in range(1, len(mkl.dims)):
                    hp_name = 'w%d' % i
                    hyper_parameters[0][hp_name] = Real(0.0, 1.0, prior='uniform')

                # Construct the BO hyperparam search problem
                # Note that our implementation above uses spherical coordinates and hence transforms the weights
                clf = BayesSearchCV(mkl, hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring,
                                    n_iter=200, n_jobs=n_jobs, verbose=100)
            else:
                # Random grid search without BO
                # Simpler, but would be more resource intensive.
                hyper_parameters = [{
                                    'kernel_type': ['linear', 'rbf']
                                    }]

                # Add only as many random weight dimensions as needed here
                # so that random search space is kept small.
                # w3 =  [ 0, 0.01, 0.1, 0.5, 0.75, 1] etc.
                for i in range(1, len(mkl.dims)):
                    hp_name = 'w%d' % i
                    hyper_parameters[0][hp_name] = [0, 0.01, *[x / 100 for x in range(5, 100, 5)], 0.99, 1]

                clf = GridSearchCV(mkl, hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring,
                                   n_jobs=n_jobs, verbose=2)

        # Random Forest
        if method == 'rf':
            hyper_parameters = [{
                'n_estimators': [s for s in range(100, 1001, 200)],
                'max_features': ['sqrt', 'log2'],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'criterion': ['gini', 'entropy']
            }]
            clf = GridSearchCV(RandomForestClassifier(n_jobs=n_jobs, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=verbose)

        # Multi-layer Perceptron
        if method == 'mlp':
            hyper_parameters = [{
                'numHiddenLayers': [1, 2, 3],
                'epochs': [30, 50, 100, 200, 300],
                'numUnits': [10, 30, 50, 100],
                'dropout_rate': [0.1, 0.3]
            }]
            model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=self.X_train.shape[1], verbose=0)
            clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=verbose)

        # Perform CV against the AE-compressed data
        clf.fit(self.X_train, self.y_train)

        # Evaluate performance of the best model on the test set
        y_pred = clf.predict(self.X_test)
        y_prob = clf.predict_proba(self.X_test)[:, 1]

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        y_true = self.y_test
        metrics = {'AUC': round(roc_auc_score(y_true, y_prob), 4),
                   'ACC': round(accuracy_score(y_true, y_pred), 4),
                   'Recall': round(recall_score(y_true, y_pred), 4),
                   'Precision': round(precision_score(y_true, y_pred), 4),
                   'F1_score': round(f1_score(y_true, y_pred), 4),
                   'total_runtime(s)': round((time.time() - self.t_start), 2),
                   'classfication_time(s)': round((time.time() - clf_start_time), 2),
                   'best_hyperparameter': str(clf.best_params_)
                   }
        return metrics


def train_modality(Xfile, Yfile, AE_type, gradient_threshold=100, latent_dims=8, num_filters=3,
                   num_internal_layers=1, use_2D_kernels=False,
                   max_training_duration=np.inf, seed=42, classifiers_to_train=[]):
    """
    A function that trains the AE only, but not the classifier itself.

    Currently this is invoked by BOHB for parameter optimisation.

    Parameters
    ----------
    Xfile : str
        Name of the file containing features matrix (i.e. X). First row and column
        will be treated as labels and ignored. REQUIRED

    Yfile : str
        Name of the file containing the class labels. Should be a 2-column tsv.
        1st column should contain the sample identifiers, and should match those in
        Xfile. 2nd column should contain binary class of each sample. REQUIRED

    AE_type : 'AE', 'CAE', 'VAE', 'LSTM', 'GRP' or 'PCA' as str
        The type of dimensionality reduction method to be used. REQUIRED

    gradient_threshold : float
        Maximum gradient magnitude to be allowed during training. Smaller values
        reduce the likelihood of gradient explosion, however might considerably
        increase the training time. Not used by default.

    latent_dims : int or array of ints
        Number of dimensions of the latent space. 8 by default and only used by
        AE and VAE. To include hidden layer(s), set latent_dims to a list of ints

    num_filters : int
        Number of kernels to be trained. CAE only, 3 by default.

    num_internal_layers : int
        Number of layers between the input and latent layer. CAE only, 1 by default.

    use_2D_kernels : bool
        Whether to convert the feature vector into a square matrix and train 2D
        kernels. The default behaviour is to keep as is and train 1D kernels.

    max_training_duration : float
        Maximum duration to allow during the AE training, in seconds. If none
        provided, there is no time limit.

    seed : int
        Seed for the random number generator. Defaults to 42.

    classifiers_to_train : list of str
        Optional list of names of the binary classifiers to train. None will be
        trained by default.
    """

    # create an object and load data
    # Each different experimental component needs to be treated as 1 separate Modality.
    m = Modality(Xfile=Xfile, Yfile=Yfile, seed=seed, clipnorm_lim=gradient_threshold, max_training_duration=max_training_duration)

    # load data into the object
    m.load_X_data()

    # time check after data has been loaded
    m.t_start = time.time()

    # Preprocess dimensions: single int vs. list of ints
    if type(latent_dims) == int:
        latent_dims = [latent_dims]

    # Representation learning (Dimensionality reduction)
    AE_type = AE_type.upper()
    if AE_type in ['AE', 'SAE', 'DAE']:
        m.sae(dims=latent_dims, loss='mse', verbose=0)
    elif AE_type == 'CAE':
        m.cae(num_internal_layers=num_internal_layers, loss='mse', num_filters=num_filters,
              use_2D=use_2D_kernels, verbose=0)
    elif AE_type == 'VAE':
        m.vae(dims=latent_dims, loss='mse', verbose=0)
    elif AE_type == 'LSTM':
        m.lstm(dims=latent_dims)
    elif AE_type == 'GRP':
        m.grp()
    elif AE_type == 'PCA':
        m.pca()
    else:
        raise NameError('Autoencoder type %s is not available' % AE_type)

    if len(classifiers_to_train) == 0:
        val_loss = m.get_min_loss()
        return val_loss
    else:
        # Optional training attempt of classifiers for 1 modality only.
        # Iff so, labelled data is needed.
        numFolds = 5
        scoring = 'roc_auc'  # options: 'roc_auc', 'accuracy', 'f1', 'recall', 'precision'

        # Training classification model(s)
        m.load_Y_data()

        # Support vector classifier
        if 'svm' in classifiers_to_train:
            metrics = m.classification(method='svm', cv=numFolds, scoring=scoring, cache_size=1000)

        # multi-kernel learning with SVM
        if 'mkl' in classifiers_to_train:
            metrics = m.classification(method='mkl', cv=numFolds, scoring=scoring, cache_size=1000, use_bayes_opt=False,
                                       subpart_dims=latent_dims)

        # Random forest
        if 'rf' in classifiers_to_train:
            metrics = m.classification(method='rf', cv=numFolds, scoring=scoring)

        # Multi layer perceptron
        if 'mlp' in classifiers_to_train:
            metrics = m.classification(method='mlp', cv=numFolds, scoring=scoring)

        return 1 - metrics['AUC']


if __name__ == '__main__':
    train_modality(Xfile='../dm_data/IBD_X_abundance.tsv', Yfile='../dm_data/IBD_Y.tsv', latent_dims=[7, 5], AE_type='VAE')
