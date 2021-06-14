import pandas, numpy, os, sys, umap, random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.harmae import get_data
from src.preprocessing import run_pca
from src.batch_analysis import plot_encodings_umap
from src.evaluation import find_best_epoch

if __name__ == "__main__":

    pars = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/sarahs/grid5/b85117f6/parameters_b85117f6.csv', index_col=0).T
    print(str(pars['stopped_early'].values[0]))