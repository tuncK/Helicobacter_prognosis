#!/usr/bin/env python
# Script to process all data modalities of the project independently 1 by one
# Then a classifier will be trained that makes use of all the data combined

import startBOHB
import MyWorker
import autoencoders
import pandas as pd

# List of data files to use for training
# Features should be on the rows, 1st column with feature label
# Samples should be on the columns, each column should have a common sample name
data_tables = [
                    '../preprocessed/16S.tsv',
                    # '../preprocessed/ELISA.tsv',
                    # '../preprocessed/host_transcriptome.tsv',
                    # '../preprocessed/metabolomics.tsv',
                    # '../preprocessed/mgx_enzymes.tsv',
                    # '../preprocessed/mgx_pathways.tsv',
                    '../preprocessed/mgx_species.tsv',
                    # '../preprocessed/mtx_enzymes.tsv',
                    # '../preprocessed/mtx_pathways.tsv',
                    # '../preprocessed/proteome.tsv',
                    # '../preprocessed/virome.tsv'
                  ]


# Train an AE for each modality separately.
# Find the best param configuration with BOHB without saving the model
# Then save the reduced representation of the data with best hyperparams found.
X_latent_files = []
dims = []
for data_table in data_tables:
    print('Representation learning on %s' % data_table)
    best_conf = startBOHB.start_BOHB(min_budget=30, max_budget=100, data_table=data_table, n_workers=6, n_iterations=10)

    # This time, we need to save the result (and feed into the classifier later on)
    (grad_threshold, dim, seed, AE_type) = MyWorker.extract_convert_params(best_conf)
    dims.append(dim)

    m = autoencoders.Modality(data=data_table, dims=dim, seed=seed, clipnorm_lim=grad_threshold)

    # load data into the object
    m.load_data(dtype='int64')
    
    # Decide on which AE worked the best and use.
    if AE_type in ['AE', 'SAE', 'DAE']:
        ae_func = m.ae
    elif AE_type == 'CAE':
        ae_func = m.cae
    elif AE_type == 'VAE':
        ae_func = m.vae
    else:
        raise NameError('Autoencoder type %s is not available' % AE_type)

    # Representation learning, no time limit
    ae_func(dims=[dim], loss='mse', verbose=1, save_model=True)
    latent_rep = m.get_transformed_data()

    # Retrieved AE-compressed feature set and keep in a big matrix.
    # write the learned representation of the training set as a file
    rep_file = "../results/" + m.prefix + m.data + "_rep.tsv"
    latent_rep.to_csv(rep_file, sep='\t', header=True, index=True)
    X_latent_files.append(rep_file)
    print("The learned representation of the training set has been saved in '{}'".format(rep_file))


print('Combining AE of each modality...')
latent_dfs = []
for filename in X_latent_files:
    df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
    latent_dfs.append(df)

print('Writing the combined dataset to file...')
df_combined = pd.concat(latent_dfs, axis=0, join='inner')
df_combined.columns = [x.split('.')[0] for x in list(df_combined)]
df_combined.to_csv('../results/all_latent.tsv', sep='\t', header=True, index=True)


# Final classifier
# Now train an MKL that takes all reduced representations and predicts the class labels

# Generate a fake modality containing all datasets
# We use this for classification only, no AE
m = autoencoders.Modality(data='../results/all_latent.tsv', dims=dims)
m.load_data(dtype='int64')
m.load_labels(filename='../preprocessed/class-labels.tsv', dtype='int64')

print('Starting classification with MKL...')
numFolds = 5
scoring = 'roc_auc'  # options: 'roc_auc', 'accuracy', 'f1', 'recall', 'precision'
m.classification(method='mkl', cv=numFolds, scoring=scoring, cache_size=1000, use_bayes_opt=False)

print('All done')
