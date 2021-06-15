import pandas, numpy, os, sys, umap, random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.harmae import get_data
from src.preprocessing import run_pca
from src.batch_analysis import plot_encodings_umap
from src.evaluation import find_best_epoch

import h5py

if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/res/sarahs/grid5_with_early stopping/b2a75470/normalized_b2a75470.csv'
    normalized = pandas.read_csv(path, index_col=0)

    hf = h5py.File(path.replace('.csv', '.h5'), 'w')
    hf.create_dataset('normalized', data=normalized)
    hf.close()