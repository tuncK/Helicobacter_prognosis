# Pre-process downloaded public dataset before ML
# HMP2: human metagenome project downloaded from:
# https://ibdmdb.org/tunnel/public/summary.html

import pandas as pd
import numpy as np


# Parse the metadata table and hence generate a look-up table
# to convert different patient identifiers
df = pd.read_csv("../hmp2_metadata.csv", sep=",")
extid2pid = {}
pid_2_diagnosis = {}
for pid in set(df['Participant ID']):
    this_patient = df.loc[df['Participant ID'] == pid]
    for ext_id in list(this_patient['External ID']):
        # Direct 1-1 correspondance
        extid2pid[ext_id] = pid

        # Some patient ids have _P, _TR suffixes. Also record the root in the dict.
        extid2pid[ext_id.split('_')[0]] = pid

    # Also generate a lookup table of disease states of PIDs.
    diagnoses = this_patient['diagnosis']
    if len(set(diagnoses)) != 1:
        raise Exception('No 1-1 correspondance between subject identifiers and class labels')
    else:
        diagnosis = list(diagnoses)[0]
        if diagnosis == 'CD' or diagnosis == 'UC':
            pid_2_diagnosis[pid] = 1
        else:
            pid_2_diagnosis[pid] = 0


# Write out patient id vs diagnosis, i.e. class labels
s = pd.Series(pid_2_diagnosis, name='is_IBD')
s.index.name = 'PID'
# s.reset_index()
s.to_csv("../preprocessed/class-labels.tsv", sep='\t')


# Rename with patient IDs and sort columns alphanumerically
def rename_columns(df):
    df = df.rename(columns=extid2pid)
    col_names = sorted(set(df))
    df = df[col_names]
    return df


# Some datasets are in the public DBs, even though they are empty. Remove them to save RAM.
def eliminate_empty_datasets(df, cutoff=0):
    total = df.sum(axis=0)
    return df.loc[:, total > 0]


# Filter out dimensions with negligible hit counts
def eliminate_negligible_dims(df, cutoff=0):
    max_vals = df.max(axis=1)
    return df[max_vals >= cutoff]


# A function to process DF, common steps
def preprocess_df(df, dataset_inclusion_cutoff=0, feature_inclusion_cutoff=10):
    # Normalise and filter out compounds not present in any sample appreciably.
    # Also eliminate 0-only samples
    df = eliminate_empty_datasets(df, cutoff=dataset_inclusion_cutoff)

    total = df.sum(axis=0)
    df = df * 1e6 / total  # in ppm
    df = df.apply(np.round).astype('int64')

    sums = df.sum(axis=1)
    df = df.loc[sums.sort_values(ascending=False).index]

    df = eliminate_negligible_dims(df, feature_inclusion_cutoff)
    df.loc['OTHER'] = 1e6 - df.sum(axis=0)

    return rename_columns(df)


# Scale across each feature (i.e. across rows)
# z-score is returned
def standard_scaler(df):
    # Transposes are needed because pandas cannot do the matrix - vector algebra otherwise.
    scaled_df = (df.transpose() - df.transpose().mean()) / df.transpose().std()
    return scaled_df.transpose()


def minmax_scaler(df):
    # Transposes are needed because pandas cannot do the matrix - vector algebra otherwise.
    df_t = df.transpose()
    scaled_df_t = (df_t - df_t.min()) / (df_t.max() - df_t.min())
    return scaled_df_t.transpose()


def absmax_scaler(df):
    # Transposes are needed because pandas cannot do the matrix - vector algebra otherwise.
    df_t = df.transpose()
    scaled_df_t = df_t / df_t.abs().max()
    return scaled_df_t.transpose()


####################################
# Metabolomics
####################################

if False:
    df = pd.read_csv("../proteomics/HMP2_metabolomics.csv", sep=',', index_col=6).fillna(0)

    # Retain only data columns
    selected_cols = [x for x in df.columns if x not in ['Method', 'Pooled QC sample CV', 'm/z', 'RT', 'HMDB (*Representative ID)', 'Metabolite', 'Compound']]
    df = df[selected_cols]

    df = preprocess_df(df)
    df.to_csv("../preprocessed/metabolomics.tsv", sep='\t')


# Proteome from MS/MS (?)
if False:
    df = pd.read_csv("../proteomics/HMP2_proteomics_ecs.tsv", sep='\t', index_col=0).fillna(0).astype('int64')

    df = df.drop(['UNGROUPED'])
    df = eliminate_empty_datasets(df, cutoff=10)
    df = eliminate_negligible_dims(df, cutoff=1)
    df = rename_columns(df)

    df.to_csv("../preprocessed/proteome.tsv", sep='\t')


####################################
# Host transcriptome
####################################
if False:
    df = pd.read_csv("../host_tx_counts.tsv", sep='\t', index_col=0).fillna(0)
    df = preprocess_df(df)

    df = minmax_scaler(df)
    df.to_csv("../preprocessed/host_transcriptome.tsv", sep='\t')


####################################
# Metagenomics
####################################

# 16 S
if False:
    df = pd.read_csv("../metagenome/16S_taxonomic_profiles.tsv", sep='\t', index_col=-1).fillna(0)

    df = df.drop(['#OTU ID'], axis=1).astype('int64')
    df = eliminate_empty_datasets(df, cutoff=100)
    df = eliminate_negligible_dims(df, cutoff=1)
    df = rename_columns(df)

    df.to_csv("../preprocessed/16S.tsv", sep='\t')


if False:
    filename = "../metagenome/virome_virmap_analysis.tsv"
    df = pd.read_csv(filename, sep='\t', index_col=0).fillna(0)
    df.index = [x.split('taxId=')[1] for x in df.index]
    df.index.name = 'Virus taxid'

    df = preprocess_df(df)
    df.to_csv("../preprocessed/virome.tsv", sep='\t')


if False:
    filename = "../metagenome/hmp2_mgx_taxonomy.tsv"
    df = pd.read_csv(filename, sep='\t', index_col=0).fillna(0)

    # Retain only species-level entries
    df = df.drop(['UNKNOWN'])
    species = [x for x in df.index if "|s__" in x]
    df = df.loc[species]

    # Simplify column headers
    # HSM7CZ1T_P_profile -> HSM7CZ1T
    column_map = {x: x.split("_")[0] for x in list(df)}
    df = df.rename(columns=column_map)

    df = preprocess_df(df)
    df.to_csv("../preprocessed/mgx_species.tsv", sep='\t')


def process_humann_pathways_file(infile, outfile):
    df = pd.read_csv(infile, sep='\t', index_col=0).fillna(0)

    # Retain only legit entries
    df = df.drop(['UNMAPPED'])
    to_keep = [x for x in df.index if ("UNINTEGRATED" not in x and '|unclassified' not in x)]
    df = df.loc[to_keep]

    # Eliminate 0-only feature dimensions
    # Rows are not independent dimensions, but represent levels of taxonomy tree.
    # No re-normalisation can/should be done.
    df = eliminate_empty_datasets(df, cutoff=0)
    df = eliminate_negligible_dims(df, cutoff=5)

    # Simplify column headers
    # CSM5FZ3N_P_pathabundance_cpm -> CSM5FZ3N
    column_map = {x: x.split("_")[0] for x in list(df)}
    df = df.rename(columns=column_map)

    # Rename with patient IDs and sort columns alphanumerically
    df = rename_columns(df)
    df.index.name = 'Pathway'

    # Round to int
    df = df.apply(np.round).astype('int64')
    df.to_csv(outfile, sep='\t')


process_humann_pathways_file(infile="../metagenome/hmp2_mgx_pathabundance.tsv", outfile="../preprocessed/mgx_pathways.tsv")
process_humann_pathways_file(infile="../metatranscriptome/pathabundances_3.tsv", outfile="../preprocessed/mtx_pathways.tsv")


def process_humann_enzymes_file(infile, outfile):
    df = pd.read_csv(infile, sep='\t', index_col=0, comment='#').fillna(0)

    # Retain only legit entries
    df = df.drop(['UNMAPPED'])
    to_keep = [x for x in df.index if ("UNGROUPED" not in x and '|unclassified' not in x)]
    df = df.loc[to_keep]

    # Round to int
    df = df.apply(np.round).astype('int64')

    # Eliminate 0-only samples (i.e. failures)
    # Also eliminate 0 / low count features in all samples
    # Rows are not independent dimensions, but represent levels of taxonomy tree.
    # No re-normalisation can/should be done.
    df = eliminate_empty_datasets(df, cutoff=0)
    df = eliminate_negligible_dims(df, cutoff=10)

    # Simplify column headers
    # CSM5FZ3N_P_pathabundance_cpm -> CSM5FZ3N -> C3001
    column_map = {x: x.split("_")[0] for x in list(df)}
    df = df.rename(columns=column_map)
    df = rename_columns(df)
    df.index.name = 'Enzyme'

    df.to_csv(outfile, sep='\t')


process_humann_enzymes_file(infile="../metagenome/ecs_3.tsv", outfile="../preprocessed/mgx_enzymes.tsv")
process_humann_enzymes_file(infile="../metatranscriptome/ecs_3.tsv", outfile="../preprocessed/mtx_enzymes.tsv")


####################################
# Clinical
####################################
if False:
    filename = "../hmp2_serology_Compiled_ELISA_Data.tsv"
    df = pd.read_csv(filename, sep='\t', index_col=0).fillna(0)

    # Hospital name, elisa plate position etc.
    df = df.drop(['Sample', 'Site', 'Plate'])

    # Sample identifier 220948 is not present in the metadata table. Remove?
    df = df.drop('220948', axis=1)

    df = rename_columns(df)

    df.to_csv("../preprocessed/ELISA.tsv", sep='\t')
