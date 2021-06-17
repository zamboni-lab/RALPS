import pandas, numpy, os, sys, umap, random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.harmae import get_data
from src.preprocessing import run_pca
from src.batch_analysis import plot_encodings_umap
from src.evaluation import find_best_epoch

import h5py

if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/data/sarah/'

    data = pandas.read_csv(path + 'data_with_mzs.csv')
    info = pandas.read_csv(path + 'batch_info.csv', keep_default_na=False)

    data.insert(1, 'rt', 1)
    data = data.rename(columns={'Unnamed: 0': 'name'})
    info.insert(1, 'injection.order', 1)
    info = info.rename(columns={'Unnamed: 0': 'sample.name', 'benchmark': 'class'})

    for i in range(info.shape[0]):
        if info.iloc[i, 4] == '':
            info.iloc[i, 4] = 'Subject'
        else:
            info.iloc[i, 3] = 'QC'
            info.iloc[i, 4] = 'QC'

    data.to_csv(path + 'normae_data.csv', index=False)
    info.to_csv(path + 'normae_info.csv', index=False)