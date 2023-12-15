#!/usr/bin/env python
# Script to process all data modalities of the project independently 1 by one
# Then a classifier will be trained that makes use of all the data combined

import startBOHB
import MyWorker
import autoencoders
import pandas as pd
import json


def train_1_AE(Xfile, Yfile, latent_filename):
    """
    Find the best param configuration with BOHB without saving the model
    Then save the reduced representation of the data with best hyperparams found.
    """

    # This will try many hyperparam combinations.
    # Despite using BOHB etc., many models will be trained, so might take a while.
    best_conf = startBOHB.start_BOHB(min_budget=120, max_budget=600, Xfile=Xfile,
                                     Yfile=Yfile, n_workers=1, n_iterations=10)

    # This time, we need to save the result (and feed into the classifier later on)
    best_conf = MyWorker.convert_hyper_params(best_conf)
    m = autoencoders.Modality(Xfile=Xfile, Yfile=None, **best_conf)

    # load data into the object
    m.load_X_data()

    # Decide on which AE worked the best and use.
    if best_conf['AE_type'] in ['AE', 'SAE', 'DAE']:
        ae_func = m.sae
    elif best_conf['AE_type'] == 'CAE':
        ae_func = m.cae
    elif best_conf['AE_type'] == 'VAE':
        ae_func = m.vae
    elif best_conf['AE_type'] == 'LSTM':
        ae_func = m.lstm
    else:
        raise NameError('Autoencoder type %s is not available' % best_conf['AE_type'])

    # Perform representation learning without time limit
    # with best AE type and best hyperparams
    print('Started full learning with the best AE hyperparams: %s' % best_conf)
    ae_func(loss='mse', verbose=0, save_model=True, **best_conf)

    # Write the learned representation of the training set as a file
    m.export_transformed_data(latent_filename)

    # Write the best AE hyperparams to file
    filename = latent_filename.replace('_latent.tsv', '_AEparams.json')
    with open(filename, 'w') as file:
        file.write(json.dumps(best_conf))


def combine_modalities(X_latent_files, combined_filename):
    """
    Combines the latent dimensions of each modality learnt before
    as a single table.
    """

    latent_dfs = []
    dims = []
    for filename in X_latent_files:
        df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        latent_dfs.append(df)
        dims.append(df.shape[0])

    print('Writing the combined dataset to file...')
    df_combined = pd.concat(latent_dfs, axis=0, join='inner')
    df_combined.columns = [x.split('.')[0] for x in list(df_combined)]
    df_combined.to_csv(combined_filename, sep='\t', header=True, index=True)
    return dims


# Generate a fake modality containing all datasets
# We use this for classification only, no AE
def run_MKL(combined_file, modality_dims, Yfile):
    """
    Train an MKL that takes all reduced representations and predicts the class
    labels based on all dimensions combined.
    """

    m = autoencoders.Modality(Xfile=combined_file, Yfile=Yfile)
    m.load_X_data()
    m.load_Y_data(dtype='int64')

    print('Starting classification with MKL...')
    numFolds = 5
    scoring = 'roc_auc'  # options: 'roc_auc', 'accuracy', 'f1', 'recall', 'precision'
    classifier_metrics = m.classification(method='mkl', cv=numFolds, scoring=scoring, cache_size=1000, use_bayes_opt=False,
                                          subpart_dims=modality_dims, verbose=1)
    print(classifier_metrics)
    return classifier_metrics


#########################################################################
# List of data files to use for training
# Features should be on the rows, 1st column with feature label
# Samples should be on the columns, each column should have a common sample name
data_tables = [
        '../dm_data/Cirrhosis_X_abundance.tsv',
        '../dm_data/Cirrhosis_X_marker.tsv',
        '../dm_data/Colorectal_X_abundance.tsv',
        '../dm_data/Colorectal_X_marker.tsv',
        '../dm_data/IBD_X_abundance.tsv',
        '../dm_data/IBD_X_marker.tsv',
        '../dm_data/T2D_X_abundance.tsv',
        '../dm_data/T2D_X_marker.tsv',
        '../dm_data/WT2D_X_abundance.tsv',
        '../dm_data/WT2D_X_marker.tsv'
]

data_tables = data_tables[:2]

# Names for the AE-compressed data tables
X_latent_files = ['../results/' + x.split('/')[-1].split('_')[0] + '/' + x.split('/')[-1].split('_')[-1].split('.')[0] + '_latent.tsv' for x in data_tables]

# Train an AE for each modality separately.
# The latent representations will be saved in file, and can be used later on.
for i in range(len(data_tables)):
    print('Representation learning on %s' % data_tables[i])
    train_1_AE(Xfile=data_tables[i], Yfile=None, latent_filename=X_latent_files[i])

print('All AEs were trained')


# Group the different modalities for the same dataset and traing a classifier
for i in range(0, len(X_latent_files), 2):
    # Concatenate latent dims of 2 modalities and write to a new file
    file1 = X_latent_files[i]
    file2 = X_latent_files[i+1]
    combined_filename = '/'.join(file1.split('/')[:-1]) + '/combined_latent.tsv'
    modality_dims = combine_modalities(X_latent_files=[file1, file2], combined_filename=combined_filename)

    # Train MKL on the file with all latent dims combined.
    dataset_name = file1.split('/')[-2]
    Yfile = '../dm_data/' + dataset_name + '_Y.tsv'
    run_MKL(combined_file=combined_filename, modality_dims=modality_dims, Yfile=Yfile)

print('All done')
