import pandas, numpy, os, sys, umap, random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from src.ralps import get_data
# from src.preprocessing import run_pca
# from src.batch_analysis import plot_encodings_umap
# from src.evaluation import find_best_epoch

import h5py
from pathlib import Path

if __name__ == "__main__":

    history = pandas.read_csv('D:\ETH\projects\\normalization\\res\\SRM+SPP\\best_models.csv')
    # history = history.iloc[:25,:]

    # grouping slice
    df = history[history['all_grouping'] <= numpy.percentile(history['all_grouping'].values, 30)]
    df2 = history[history['all_corr'] >= numpy.percentile(history['all_corr'].values, 70)]

    df = df[df['all_corr'] >= numpy.percentile(df['all_corr'].values, 70)].sort_values('rec_loss')
    df3 = df2[df2['batch_vc'] <= numpy.percentile(df['batch_vc'].values, 30)].sort_values('rec_loss')
    df2 = df2[df2['all_grouping'] <= numpy.percentile(df['all_grouping'].values, 30)].sort_values('rec_loss')

    all_best = pandas.DataFrame()
    all_best = pandas.concat([all_best, history[
        (history['batch_vc'] <= numpy.percentile(history['batch_vc'].values, 30))
        & (history['all_corr'] >= numpy.percentile(history['all_corr'].values, 70))
    ]])
    all_best = pandas.concat([all_best, history[
        (history['all_grouping'] <= numpy.percentile(history['all_grouping'].values, 30))
        & (history['all_corr'] >= numpy.percentile(history['all_corr'].values, 70))
        ]])
    all_best = all_best.drop_duplicates()

    print(df2)
    print(df2)



