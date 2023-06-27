#!/usr/bin/env python
# Script to process all data modalities of the project independently 1 by one
# Then a classifier will be trained that makes use of all the data combined

import startBOHB
import MyWorker
import autoencoders
import pandas as pd
import json

# List of data files to use for training
# Features should be on the rows, 1st column with feature label
# Samples should be on the columns, each column should have a common sample name
data_tables = [
                    '../dm_data/IBD_X_abundance.tsv',
                    '../dm_data/IBD_X_marker.tsv'
              ]

# Train an AE for each modality separately.
# Find the best param configuration with BOHB without saving the model
# Then save the reduced representation of the data with best hyperparams found.
X_latent_files = []
for data_table in data_tables:
    print('Representation learning on %s' % data_table)
    best_conf = startBOHB.start_BOHB(min_budget=30, max_budget=100, data_table=data_table, n_workers=6, n_iterations=10)

    # This time, we need to save the result (and feed into the classifier later on)
    best_conf = MyWorker.convert_hyper_params(best_conf)
    m = autoencoders.Modality(Xfile=data_table, Yfile=None, **best_conf)

    # load data into the object
    m.load_X_data()

    # Decide on which AE worked the best and use.
    if best_conf['AE_type'] in ['AE', 'SAE', 'DAE']:
        ae_func = m.ae
    elif best_conf['AE_type'] == 'CAE':
        ae_func = m.cae
    elif best_conf['AE_type'] == 'VAE':
        ae_func = m.vae
    else:
        raise NameError('Autoencoder type %s is not available' % best_conf['AE_type'])

    # Representation learning, no time limit
    print('Started full learning with best AE hyperparams: %s' % best_conf)
    ae_func(loss='mse', verbose=0, save_model=True, **best_conf)

    # Write the learned representation of the training set as a file
    filename = '../results/' + data_table.split('/')[-1].split('.')[0] + '_latent.tsv'
    m.export_transformed_data(filename)
    X_latent_files.append(filename)

    # Write the best AE hyperparams to file
    filename = '../results/' + data_table.split('/')[-1].split('.')[0] + '_AE_params.json'
    with open(filename, 'w') as file:
        file.write(json.dumps(best_conf))

print('Combining AE of each modality...')
latent_dfs = []
dims = []
for filename in X_latent_files:
    df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
    latent_dfs.append(df)
    dims.append(df.shape[0])

print('Writing the combined dataset to file...')
df_combined = pd.concat(latent_dfs, axis=0, join='inner')
df_combined.columns = [x.split('.')[0] for x in list(df_combined)]
df_combined.to_csv('../results/all_latent.tsv', sep='\t', header=True, index=True)


# Final classifier
# Now train an MKL that takes all reduced representations and predicts the class labels

# Generate a fake modality containing all datasets
# We use this for classification only, no AE
m = autoencoders.Modality(Xfile='../results/all_latent.tsv', Yfile='../dm_data/IBD_Y.tsv')
m.load_X_data()
m.load_Y_data(dtype='int64')

print('Starting classification with MKL...')
numFolds = 5
scoring = 'roc_auc'  # options: 'roc_auc', 'accuracy', 'f1', 'recall', 'precision'
classifier_metrics = m.classification(method='mkl', cv=numFolds, scoring=scoring, cache_size=1000, use_bayes_opt=False,
                                      subpart_dims=dims, verbose=1)
print(classifier_metrics)

print('All done')
