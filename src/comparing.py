
import numpy, pandas, seaborn, time
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

from src.models.ae import get_data
from src.batch_analysis import plot_batch_cross_correlations, compute_cv_for_samples_types
from src.batch_analysis import compute_number_of_clusters_with_hdbscan
from src.utils import combat
from src.constants import benchmark_sample_types as benchmarks
from src.constants import regularization_sample_types as regs
from src.constants import data_path as path
from src.constants import path_to_other_methods, path_to_my_best_method, user, batches
from src.batch_analysis import get_sample_cross_correlation_estimate


def plot_benchmarks_corrs_for_methods(methods=['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'my_best']):

    save_to = '/Users/{}/ETH/projects/normalization/res/other_methods/plots/'.format(user)
    for method in methods:
        plot_correlations_for_normalization(method, save_to)
    print('benchmark correlations saved')


def plot_correlations_for_normalization(method, save_to):

    if method == 'none':
        data = get_data(shuffle=False)
        normalized = data.iloc[:, 1:]
    elif method == 'my_best':
        # hardcode
        normalized = pandas.read_csv(path_to_my_best_method, index_col=0)
    else:
        normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

    plot_batch_cross_correlations(normalized, method, '', sample_types_of_interest=benchmarks, save_to=save_to)


def plot_benchmarks_cvs_for_methods(methods=['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'my_best']):

    all_cvs = pandas.DataFrame()

    for method in methods:

        if method == 'none':
            data = get_data(shuffle=False)
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best_method, index_col=0)
        else:
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

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
        ax.tick_params(labelrotation=45)

    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig('/Users/{}/ETH/projects/normalization/res/other_methods/plots/cvs.pdf'.format(user))
    print('variation coefs saved')


def plot_samples_corrs_for_methods(methods=['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'my_best']):

    corrs = []
    for method in methods:

        if method == 'none':
            data = get_data(shuffle=False)
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best_method, index_col=0)
        else:
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

        res = get_sample_cross_correlation_estimate(normalized, sample_types_of_interest=regs)
        corrs.append(res)

    res = pandas.DataFrame({'method': methods, 'corr': corrs})

    seaborn.barplot(x='method', y='corr', data=res)
    pyplot.xlabel('Normalization')
    pyplot.ylabel('Correlation sum')
    pyplot.grid()
    pyplot.tick_params(labelrotation=45)
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig('/Users/{}/ETH/projects/normalization/res/other_methods/plots/corrs.pdf'.format(user))
    print('overall correlations saved')


def get_grouping_coefs_for_samples(method, clustering, total_clusters):

    grouping_coefs = {}
    coefs_sum = 0
    for sample in benchmarks:
        n_sample_clusters = len(set(clustering[sample]))
        max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
        coef = (n_sample_clusters - 1) / max_n_clusters
        grouping_coefs[sample] = coef
        coefs_sum += coef

    # print('{}: mean coef: {}'.format(method, coefs_sum / len(grouping_coefs)))
    return grouping_coefs


def plot_benchmarks_grouping_coefs_for_methods(methods=['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'my_best']):

    data = get_data()
    pars = {'latent_dim': data.shape[1], 'n_batches': 7, 'n_replicates': 3}

    all_grouping_coefs = pandas.DataFrame()

    for method in methods:

        if method == 'none':
            normalized = data

        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best_method, index_col=0)
            normalized['batch'] = data['batch']
        else:
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)
            normalized['batch'] = data['batch']

        clustering, total_clusters = compute_number_of_clusters_with_hdbscan(normalized, pars, print_info=False, sample_types_of_interest=benchmarks)
        grouping_dict = get_grouping_coefs_for_samples(method, clustering, total_clusters)

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
        ax.tick_params(labelrotation=45)

    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig('/Users/{}/ETH/projects/normalization/res/other_methods/plots/grouping_coefs.pdf'.format(user))
    print('grouping coefs saved')


def plot_normalized_spectra_for_methods(file_ext='pdf', methods=['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'my_best']):

    save_to = '/Users/{}/ETH/projects/normalization/res/other_methods/plots/'.format(user)
    mz = pandas.read_csv('/Users/{}/ETH/projects/normalization/data/filtered_data.csv'.format(user))['mz'].values
    color_dict = {'0108': 'k', '0110': 'g', '0124': 'r', '0219': 'c', '0221': 'm', '0304': 'y', '0306': 'b'}

    for method in methods:

        if method == 'none':
            data = get_data(shuffle=False).drop(columns=['batch'])
        elif method == 'my_best':
            data = pandas.read_csv(path_to_my_best_method, index_col=0)
        else:
            data = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

        samples = data.index
        normalized = data.values

        scaled = (normalized - numpy.mean(normalized)) / numpy.std(normalized)

        pyplot.figure(figsize=(8, 6))
        for i in range(scaled.shape[0]):
            batch_id = [batch in samples[i] for batch in batches].index(True)
            color = color_dict[batches[batch_id]]

            if color == 'b':
                pyplot.plot(mz, scaled[i, :], '{}o'.format(color), alpha=0.2)
            else:
                pyplot.plot(mz, scaled[i, :], '{}o'.format(color), alpha=0.4)

        pyplot.ylim(bottom=0, top=50)
        pyplot.title(method)
        pyplot.xlabel('mz')
        pyplot.ylabel('scaled normalized intensities')
        pyplot.grid()
        # pyplot.show()
        pyplot.savefig(save_to + 'spectra_{}.{}'.format(method, file_ext))
    print('spectral patterns saved')


def check_relevant_intensities_for_methods(methods=['combat', 'eigenMS', 'waveICA', 'my_best']):
    """ Methods 'lev+eig', 'pqn+pow' are excludede by default, since they don't output intensities. """

    for method in methods:
        if method == 'none':
            data = get_data(shuffle=False)
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best_method, index_col=0)
        else:
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

        print("method: {}".format(method))
        print('negative: {}%'.format(
            round(100 * (normalized < 0).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print('< 1000: {}%'.format(
            round(100 * (normalized < 1000).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print()


if __name__ == "__main__":

    # benchmarks
    plot_benchmarks_cvs_for_methods()
    plot_benchmarks_grouping_coefs_for_methods()
    plot_benchmarks_corrs_for_methods()

    # all samples
    plot_normalized_spectra_for_methods(file_ext='png')
    plot_samples_corrs_for_methods()  # not very informative
    check_relevant_intensities_for_methods()
