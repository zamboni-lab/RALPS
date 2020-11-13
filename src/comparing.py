
import numpy, pandas, seaborn
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

    for method in ['no', 'combat']:

        if method == 'no':
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


def plot_grouping_coefs_for_methods():

    data = get_data()

    combat_normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])

    pars = {'latent_dim': data.shape[1], 'n_batches': 7, 'n_replicates': 3}

    numpy.random.seed(5)
    clustering, total_clusters = compute_number_of_clusters_with_hdbscan(data, pars, print_info=False,
                                                                         sample_types_of_interest=benchmarks)
    grouping_coefs = []
    for sample in benchmarks:
        n_sample_clusters = len(set(clustering[sample]))
        max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
        coef = (n_sample_clusters - 1) / max_n_clusters
        grouping_coefs.append(coef)
        print(coef)
    b_grouping = numpy.mean(grouping_coefs)
    print('original data: {}'.format(b_grouping))

    # use combat to normalize and cluster
    combat_normalized = combat_normalized.T
    combat_normalized['batch'] = data['batch']
    clustering, total_clusters = compute_number_of_clusters_with_hdbscan(combat_normalized, pars, print_info=False,
                                                                         sample_types_of_interest=benchmarks)
    grouping_coefs = []
    for sample in benchmarks:
        n_sample_clusters = len(set(clustering[sample]))
        max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
        coef = (n_sample_clusters - 1) / max_n_clusters
        grouping_coefs.append(coef)
    b_grouping = numpy.mean(grouping_coefs)
    print('combat: {}'.format(b_grouping))

    # TODO: wrap into a function, plot bars

    pass


if __name__ == "__main__":
    plot_cvs_for_methods()