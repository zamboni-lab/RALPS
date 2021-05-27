import pandas, numpy, os, sys, umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.harmae import get_data
from src.preprocessing import run_pca
from src.batch_analysis import plot_encodings_umap

if __name__ == "__main__":

    # config = {'data_path': '/Users/andreidm/ETH/projects/normalization/data/filtered_data_v4.csv',
    #           'info_path': '/Users/andreidm/ETH/projects/normalization/data/batch_info_v4.csv',
    #           'min_relevant_intensity': 1000}
    #
    # data = get_data(config)
    # batch = data['batch']
    # data = data.iloc[:, 1:]

    data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/P2_SRM_0001+P2_SPP_0001/da9e81db/normalized_da9e81db.csv', index_col=0)

    transformer = PCA(n_components=50)
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    pca_reduced = pandas.DataFrame(transformer.fit_transform(scaled_data))
    pca_reduced.insert(0, 'batch', 0)

    parameters = {'n_batches': 7, 'n_replicates': 3, 'id': 'test'}
    plot_encodings_umap(pca_reduced, 'pca_normalized_data', parameters)





    print()