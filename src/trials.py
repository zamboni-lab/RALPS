import pandas, numpy, os, sys, umap
from src.preprocessing import run_pca

if __name__ == "__main__":

    path = '/Users/dmitrav/ETH/projects/normalization/res/activations/'

    recs = []
    zero_percents = []
    for file in os.listdir(path):
        if not file.startswith('.'):
            rec = pandas.read_csv(path+file, index_col=0)

            percent = (rec < 0).sum().sum() / rec.shape[0] / rec.shape[1]
            zero_percents.append(percent)

    for i, percent in enumerate(zero_percents):
        print(i, percent)


