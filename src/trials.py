import pandas, numpy, os, sys
from src.preprocessing import run_pca

if __name__ == "__main__":

    data = pandas.read_csv('/Users/dmitrav/ETH/projects/normalization/data/filtered_data.csv')

    values = data.T.values[3:, :]

    for n in range(40, 50):

        print("N={}".format(n))
        scaled, transformer = run_pca(values, n=n)
        print('total ratio of variance explained: {}'.format(sum(transformer.explained_variance_ratio_)))
        print()

