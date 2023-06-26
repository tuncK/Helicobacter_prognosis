# Merges multiple biom files into a single consolidated tsv.
# Entries are merged if they share a common index.
# Else, new columns are generated

# You can convert biom -> tsv by using:
# biom convert -i input.biom -o output.tsv --to-tsv --header-key taxonomy

import os
import pandas as pd


dirname = "./sweedish twins 16S/"
dataframes = []
for file in os.listdir(dirname):
    filename = os.fsdecode(file)
    if filename.endswith(".tsv"):
        print(filename)
        df = pd.read_csv(dirname + filename, sep='\t', index_col=0, skiprows=1)
        df = df.iloc[:, 0]
        dataframes.append(df)

print('All files were read')

df = pd.concat(dataframes, axis=1).fillna(0)
df.to_csv("swedish_twins_16S.tsv", sep='\t')
