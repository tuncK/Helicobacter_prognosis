#!/usr/bin/env python

# improved version of DeepMicro
# https://www.nature.com/articles/s41598-020-63159-5
# https://github.com/minoh0201/DeepMicro


# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam

# importing util libraries
import os
import time

# importing custom library
import DNN_models


# Fix a np.random.seed for reproducibility in numpy processing
np.random.seed(42)


class Modality(object):
    """
    Auto-encoder object for a stand-alone data modality

    Parameters
    ----------
    data : str
        Filename containing the entire training dataset
        File should contain an ndarray of shape (`n_features`,`n_samples`)

    dims : int or ndarray of shape (`n_layers`)
        Number of dimensions to include in the latent layer (and other intermediate layers)

    clipnorm_lim : float
        Threshold for gradient normalisation for numerical stability during training.
        If the norm of the gradient exceeds this threshold, it will be scaled down to this value.
        The lower is the more stable, at the expense of increased the training duration.

    seed : Seed for the number random generator. Defaults to 0.
    """

    def __init__(self, data, dims, clipnorm_lim=1, seed=0):
        self.t_start = time.time()
        self.filename = str(data)
        self.data = self.filename.split('/')[-1].split('.')[0]
        self.seed = seed
        self.dims = dims
        self.prefix = ''
        self.representation_only = False
        self.clipnorm_lim = clipnorm_lim
        self.dataset_ids = None

        params = self.data + '_' + str(dims) + '_' + str(self.seed) + '_' + str(clipnorm_lim)
        self.modelName = '%s' % params

        self.output_dir = '../results/'
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def load_data(self, dtype=None):
        """
            Read training data (X) from file
        """

        # read file
        if os.path.isfile(self.filename):
            raw = pd.read_csv(self.filename, sep='\t', index_col=0, header=0)
        else:
            raise FileNotFoundError("File {} does not exist".format(self.filename))

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

    def load_labels(self, filename, dtype=None):
        """
        Reads class labels (Y) from file
        """

        if os.path.isfile(filename):
            imported_labels = pd.read_csv(filename, sep='\t', index_col=0, header=0)

            # There might be duplicate measurements for the same patient.
            # i.e., some patient identifiers might need to be repeated.
            # The order of params also needs to match to the training data X
            labels = imported_labels.loc[self.dataset_ids]
        else:
            raise FileNotFoundError("{} does not exist".format(filename))

        # Label data validity check
        if not labels.values.shape[1] > 1:
            label_flatten = labels.values.reshape((labels.values.shape[0]))
        else:
            raise IndexError('The label file contains more than 1 column.')

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train.astype(dtype),
                                                                                label_flatten.astype('int'), test_size=0.2,
                                                                                random_state=self.seed,
                                                                                stratify=label_flatten)
        self.printDataShapes()

    def get_transformed_data(self):
        """
            Get a dataframe with the latent representation of the training data (X) to file
        """

        out = pd.DataFrame(self.X_train).transpose()
        out.columns = self.dataset_ids
        return out

    # Shallow Autoencoder & Deep Autoencoder
    def ae(self, dims=[50], epochs=10000, batch_size=100, verbose=2, loss='mean_squared_error', latent_act=False, output_act=False, act='relu', patience=20, val_rate=0.2, max_training_duration=np.inf, save_model=False):

        """
        Train the auto-encoder

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
            Parameter to watch the loss by, dafaults to mean_squared_error. Refer to keras docs.

        latent_act : bool
            Whether to use activation function for the latent layer. False by default.

        output_act : bool
            Whether to use activation function for the output layer. False by default.

        patience : int
            20 by default

        val_rate : float
            0.2 by default

        max_training_duration : int
            Maximum duration to be allowed for the training step to take. Walltime measured in seconds,
            by default the training duration is unlimited.

        save_model : bool
            Whether to save the model parameters during training, false by default. Enabling during
            hyperparameter tuning might slow down the operations. Enabling during parallel execution
            might also result in a race condition if target file names are not distinct.
        """

        # manipulating an experiment identifier in the output file
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

        # Custom keras callback function to limit total training time
        # This is needed for early stopping the procedure during BOHB
        class TimeLimit_Callback(keras.callbacks.Callback):
            def __init__(self, verbose=False):
                self.training_start_time = time.time()
                self.verbose = verbose

            def on_epoch_end(self, epoch, logs={}):
                duration = time.time() - self.training_start_time
                if self.verbose:
                    print('%ds passed so far' % duration)

                if duration >= max_training_duration:
                    print('Training exceeded time limit (max=%ds), stopping...' % max_training_duration)
                    self.model.stop_training = True
                    self.stopped_epoch = epoch

        # callbacks for each epoch
        # Disabling model saving improves the execution speed, especially if model size is big
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     TimeLimit_Callback()]

        # Exports the model to file at each iteration.
        # Due to early stopping, the final model is not necessarily the best model.
        # Constant disk IO may slow down the training considerably.
        if save_model:
            model_out_file = self.output_dir + '/' + self.modelName + '.h5'
            model_write_callback = ModelCheckpoint(model_out_file, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
            callbacks.append(model_write_callback)

            # clean up model checkpoint before use
            if os.path.isfile(model_out_file):
                os.remove(model_out_file)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train, test_size=val_rate, random_state=self.seed, stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.autoencoder, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
        self.autoencoder.summary()

        # compile model
        customised_adam = Adam(clipnorm=self.clipnorm_lim)
        self.autoencoder.compile(optimizer=customised_adam, loss=loss)

        # fit model
        self.history = self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                            verbose=verbose, validation_data=(X_inner_test, X_inner_test))

        # save loss progress
        self.saveLossProgress()

        if save_model:
            # load best model
            self.autoencoder = load_model(model_out_file)
            layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
            self.encoder = (Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output))

            # applying the learned encoder to the entire training and the test sets.
            self.X_train = self.encoder.predict(self.X_train)
            self.X_test = self.encoder.predict(self.X_test)

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
    def classification(self, method='svm', cv=5, scoring='roc_auc', n_jobs=-1, cache_size=10000, use_bayes_opt=False):
        """
        Train the classifier.

        For high dimensional data, should be run after the autoencoder is trained and
        the dimensionality is reduced.

        Parameters
        ----------
        method : str, 'svm', 'rf' or 'mlp'
            Classifier type to be trained. SVM bvy default.

        cv : int
            Number of folds for the cross-validation. Defaults to 5.

        scoring : str
            Objective function to be used to assess training quality. AUC of ROC curve by default.
        """

        clf_start_time = time.time()
        print("# Tuning hyper-parameters")

        # Support Vector Machine
        if method == 'svm':
            hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                                {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)], 'kernel': ['rbf']}]
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)

        # SVM, but with multiple kernels linearly combined
        if method == 'mkl':
            from multi_kernel_SVM import multi_kernel_SVM
            mkl = multi_kernel_SVM(probability=True, cache_size=cache_size, dims=self.dims)

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
                    hyper_parameters[0][hp_name] = [0, 0.01, *[x/100 for x in range(5, 100, 5)], 0.99, 1]

                clf = GridSearchCV(mkl, hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring,
                                   n_jobs=n_jobs, verbose=100)

        # Random Forest
        if method == 'rf':
            hyper_parameters = [{
                                'n_estimators': [s for s in range(100, 1001, 200)],
                                'max_features': ['sqrt', 'log2'],
                                'min_samples_leaf': [1, 2, 3, 4, 5],
                                'criterion': ['gini', 'entropy']
                                }]
            clf = GridSearchCV(RandomForestClassifier(n_jobs=n_jobs, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)

        # Multi-layer Perceptron
        if method == 'mlp':
            hyper_parameters = [{
                                 'numHiddenLayers': [1, 2, 3],
                                 'epochs': [30, 50, 100, 200, 300],
                                 'numUnits': [10, 30, 50, 100],
                                 'dropout_rate': [0.1, 0.3],
                                 }]
            model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=self.X_train.shape[1], verbose=0)
            clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)

        # Perform CV against the AE-compressed data
        clf.fit(self.X_train, self.y_train)

        print("Best parameter set found on the development set:")
        print(clf.best_params_)

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
        print(metrics)
        return metrics


# A function that trains the AE only, but not the classifier itself.
# Currently this is invoked by BOHB for parameter optimisation.
def train_modality(data_table='../preprocessed/16S.tsv', gradient_threshold=100, latent_dims=8, max_training_duration=np.inf, seed=42, classifiers_to_train=[]):
    # create an object and load data
    # Each different experimental component needs to be treated as 1 separate Modality.
    m = Modality(data=data_table, dims=latent_dims, seed=seed, clipnorm_lim=gradient_threshold)

    # load data into the object
    m.load_data(dtype='int64')

    # time check after data has been loaded
    m.t_start = time.time()

    # Representation learning (Dimensionality reduction)
    m.ae(dims=[latent_dims], loss='mse', max_training_duration=max_training_duration, verbose=0)

    # OPTIONAL CLASSIFIER for 1 modality only
    if len(classifiers_to_train) > 0:
        # Optional training attempt of classifiers. Iff so, labelled data is needed.
        numFolds = 5
        scoring = 'roc_auc'  # options: 'roc_auc', 'accuracy', 'f1', 'recall', 'precision'

        # Training classification model(s)
        m.load_labels(filename='../preprocessed/class-labels.tsv', dtype='int64')

        # Support vector classifier
        if 'svm' in classifiers_to_train:
            m.classification(method='svm', cv=numFolds, scoring=scoring, cache_size=1000)

        # multi-kernel learning with SVM
        if 'mkl' in classifiers_to_train:
            m.classification(method='mkl', cv=numFolds, scoring=scoring, cache_size=1000, use_bayes_opt=False)

        # Random forest
        if 'rf' in classifiers_to_train:
            m.classification(method='rf', cv=numFolds, scoring=scoring)

        # Multi layer perceptron
        if 'mlp' in classifiers_to_train:
            m.classification(method='mlp', cv=numFolds, scoring=scoring)

    val_loss = m.get_min_loss()
    return val_loss


if __name__ == '__main__':
    train_modality()
