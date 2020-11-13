
import numpy, pandas, seaborn, time
from matplotlib import pyplot

from src.models.ae import get_data
from src.batch_analysis import plot_batch_cross_correlations, compute_cv_for_samples_types
from src.batch_analysis import compute_number_of_clusters_with_hdbscan
from src.utils import combat
from src.constants import samples_with_strong_batch_effects as benchmarks
from src.constants import data_path as path


def plot_correlations_for_normalization(method):
    data = get_data()

    if method == 'combat':
        normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])
    else:
        raise ValueError('Normalization method not specified')

    plot_batch_cross_correlations(normalized.T, method, '', sample_types_of_interest=benchmarks)


def plot_cvs_for_methods():

    data = get_data()

    all_cvs = pandas.DataFrame()

    for method in ['none', 'combat']:

        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'combat':
            normalized = combat.combat(data.iloc[:, 1:].T, data['batch']).T
        else:
            raise ValueError('Normalization method not recognized')

        res = compute_cv_for_samples_types(normalized, sample_types_of_interest=benchmarks)
        res = pandas.DataFrame({'method': [method for x in range(len(res))],
                                'sample': list(res.keys()),
                                'cv': [res[key] for key in res.keys()]})
        all_cvs = pandas.concat([all_cvs, res])

    # save all on one figure
    pyplot.figure(figsize=(12, 8))

    for i, sample in enumerate(benchmarks):

        df = all_cvs.loc[all_cvs['sample'] == sample, :]

        ax = pyplot.subplot(2, 3, i + 1)
        seaborn.barplot(x='method', y='cv', data=df)
        ax.set_xlabel('Normalization')
        ax.set_ylabel('Variation coefficient')
        ax.set_title(sample)
        ax.grid(True)

    pyplot.tight_layout()
    pyplot.show()


def get_grouping_coefs_for_samples(method, clustering, total_clusters):

    grouping_coefs = {}
    coefs_sum = 0
    for sample in benchmarks:
        n_sample_clusters = len(set(clustering[sample]))
        max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
        coef = (n_sample_clusters - 1) / max_n_clusters
        grouping_coefs[sample] = coef
        coefs_sum += coef

    print('{}: mean coef: {}'.format(method, coefs_sum / len(grouping_coefs)))
    return grouping_coefs


def plot_grouping_coefs_for_methods():

    data = get_data()
    pars = {'latent_dim': data.shape[1], 'n_batches': 7, 'n_replicates': 3}

    all_grouping_coefs = pandas.DataFrame()

    for method in ['none', 'combat']:

        if method == 'none':
            normalized = data

        elif method == 'combat':
            normalized = combat.combat(data.iloc[:, 1:].T, data['batch']).T
            normalized['batch'] = data['batch']
        else:
            raise ValueError('Normalization method not recognized')

        clustering, total_clusters = compute_number_of_clusters_with_hdbscan(normalized, pars, print_info=False, sample_types_of_interest=benchmarks)
        grouping_dict = get_grouping_coefs_for_samples('none', clustering, total_clusters)

        res = pandas.DataFrame({'method': [method for x in range(len(grouping_dict))],
                                'sample': list(grouping_dict.keys()),
                                'grouping': [grouping_dict[key] for key in grouping_dict.keys()]})
        all_grouping_coefs = pandas.concat([all_grouping_coefs, res])

    # save all on one figure
    pyplot.figure(figsize=(12, 8))

    for i, sample in enumerate(benchmarks):
        df = all_grouping_coefs.loc[all_grouping_coefs['sample'] == sample, :]

        ax = pyplot.subplot(2, 3, i + 1)
        seaborn.barplot(x='method', y='grouping', data=df)
        ax.set_xlabel('Normalization')
        ax.set_ylabel('HDBSCAN grouping coef')
        ax.set_title(sample)
        ax.grid(True)

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    plot_grouping_coefs_for_methods()