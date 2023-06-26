#!/usr/bin/env python

# Script to pre-process the deepmicro DBs and re-export
# The outputs are:
# X: Super-simplified X-table with header, index row and matrix.
# Y: 1 file for all datasets containing nx1 binary vector. 1 if diseased, else 0.

import pandas as pd


# We will co-analyse the abundance and marker datasets for each disease type
# so 2 X and 1 Y per disease as output
for dataset in ['Cirrhosis', 'Colorectal', 'IBD', 'T2D', 'WT2D']:
    print(dataset)

    # Marker X
    df = pd.read_csv('./data/marker_' + dataset + '.tsv', sep='\t', header=2, index_col=0)
    features_to_select = [x for x in df.index if x.startswith('gi|')]
    features = df.loc[features_to_select]
    features.to_csv("./dm_data/" + str(dataset) + '_X_marker.tsv', sep='\t')

    # Abundance X
    df = pd.read_csv('./data/abundance_' + dataset + '.tsv', sep='\t', header=2, index_col=0)
    features_to_select = [x for x in df.index if x.startswith('k__')]
    features = df.loc[features_to_select]
    features.to_csv("./dm_data/" + str(dataset) + '_X_abundance.tsv', sep='\t')

    # Abundance Y
    disease_status = df.loc['disease']
    disease_status[disease_status != 'n'] = 1  # diseased
    disease_status[disease_status == 'n'] = 0  # healthy
    disease_status.to_csv('./dm_data/' + str(dataset) + '_Y.tsv', sep='\t')


print('All done')
