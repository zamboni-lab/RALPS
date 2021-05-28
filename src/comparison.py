
import numpy, pandas, seaborn, time
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

from src.models.ae import get_data
from src import harmae
from src.constants import default_parameters_values
from src.batch_analysis import plot_batch_cross_correlations, compute_cv_for_samples_types
from src.batch_analysis import compute_number_of_clusters_with_hdbscan
from src.batch_analysis import get_sample_cross_correlation_estimate
from src.utils import combat

# constants
user = 'andreidm'
path = '/Users/{}/ETH/projects/normalization/data/'.format(user)
batches = ['0108', '0110', '0124', '0219', '0221', '0304', '0306']

# TODO: go through the code and refactor


def plot_benchmarks_corrs_for_methods(scenario=1, save_plot=False):
    """ This method plots heatmaps of batch cross-correlations for benchmark samples. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    for method in methods:

        data = harmae.get_data({'data_path': path + 'filtered_data_v4.csv',
                                'info_path': path + 'batch_info_v4.csv',
                                'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

        _, benchmarks = harmae.extract_reg_types_and_benchmarks(data)

        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]  # keep the ordering
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        plot_batch_cross_correlations(normalized, method, '', sample_types_of_interest=benchmarks, save_to=save_to, save_plot=save_plot)

    if save_plot:
        print('benchmark correlations saved')


def plot_benchmarks_cvs_for_methods(scenario=1, save_plot=False):
    """ Plots variation coefs for benchmarks. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    all_cvs = pandas.DataFrame()
    data = harmae.get_data({'data_path': path + 'filtered_data_v4.csv',
                            'info_path': path + 'batch_info_v4.csv',
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    _, benchmarks = harmae.extract_reg_types_and_benchmarks(data)

    for method in methods:

        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]  # keep the ordering
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        res = compute_cv_for_samples_types(normalized, benchmarks)
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

    if save_plot:
        pyplot.savefig(save_to + 'cvs.pdf')
        print('variation coefs saved')
    else:
        pyplot.show()


def plot_samples_corrs_for_methods(scenario=1, save_plot=False):

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    corrs = []
    data = get_data(shuffle=False)

    for method in methods:

        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        res = get_sample_cross_correlation_estimate(normalized, regs)
        corrs.append(res)

    res = pandas.DataFrame({'method': methods, 'corr': corrs})

    seaborn.barplot(x='method', y='corr', data=res)
    pyplot.xlabel('Normalization')
    pyplot.ylabel('Correlation sum')
    pyplot.grid()
    pyplot.tick_params(labelrotation=45)
    pyplot.tight_layout()

    if save_plot:
        pyplot.savefig(save_to + 'corrs.png')
        print('overall correlations saved')
    else:
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

    # print('{}: mean coef: {}'.format(method, coefs_sum / len(grouping_coefs)))
    return grouping_coefs


def get_paths_and_methods(scenario):

    if scenario == 1:
        methods = ['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'normAE', 'my_best']
        path_to_my_best = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/best_model/2d48bfb2_best/normalized_2d48bfb2.csv'
        path_to_others = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/other_methods/'
        save_to = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/other_methods/plots/'
    elif scenario == 2:
        methods = ['none', 'normAE', 'my_best']
        path_to_my_best = path_to_my_best_method_2
        path_to_others = path_to_other_methods_2
        save_to = '/Users/andreidm/ETH/projects/normalization/res/fake_reference_samples/other_methods/plots/'.format(user)
    else:
        raise ValueError("Indicate application scenario: 1 or 2.")

    return methods, path_to_my_best, path_to_others, save_to


def plot_benchmarks_grouping_coefs_for_methods(scenario=1, save_plot=False):

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    data = harmae.get_data({'data_path': path + 'filtered_data_v4.csv',
                            'info_path': path + 'batch_info_v4.csv',
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    _, benchmarks = harmae.extract_reg_types_and_benchmarks(data)

    pars = {'latent_dim': data.shape[1]-1,
            'n_batches': len(data['batch'].unique()),
            'n_replicates': default_parameters_values['n_replicates']}

    all_grouping_coefs = pandas.DataFrame()

    for method in methods:

        if method == 'none':
            normalized = data
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
            normalized['batch'] = data['batch']
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]  # keep the ordering as inn other datasets
            normalized['batch'] = data['batch']
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            normalized['batch'] = data['batch']

        clustering, total_clusters = compute_number_of_clusters_with_hdbscan(normalized, pars, benchmarks, print_info=False)
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

    if save_plot:
        pyplot.savefig(save_to + 'grouping_coefs.pdf')
        print('grouping coefs saved')
    else:
        pyplot.show()


def plot_normalized_spectra_for_methods(scenario=1, file_ext='pdf', save_plot=False):

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    mz = pandas.read_csv('/Users/{}/ETH/projects/normalization/data/filtered_data.csv'.format(user))['mz'].values
    color_dict = {'0108': 'k', '0110': 'g', '0124': 'r', '0219': 'c', '0221': 'm', '0304': 'y', '0306': 'b'}

    data = get_data(shuffle=False)

    for method in methods:

        if method == 'none':
            normalized = data.drop(columns=['batch'])
        elif method == 'my_best':
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]  # keep the ordering as inn other datasets
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        samples = normalized.index
        normalized = normalized.values

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

        if save_plot:
            pyplot.savefig(save_to + 'spectra_{}.{}'.format(method, file_ext))
        else:
            pyplot.show()

    if save_plot:
        print('spectral patterns saved')


def check_relevant_intensities_for_methods(scenario=1):
    """ Methods 'lev+eig', 'pqn+pow', 'normAE' are excluded by default, since they don't output intensities. """

    if scenario == 1:
        methods = ['none', 'combat', 'eigenMS', 'waveICA', 'my_best']
        path_to_my_best = path_to_my_best_method_1
        path_to_others = path_to_other_methods_1
    elif scenario == 2:
        methods = ['none', 'my_best']
        path_to_my_best = path_to_my_best_method_2
        path_to_others = path_to_other_methods_2
    else:
        raise ValueError("Please, indicate application scenario.")

    data = get_data(shuffle=False)
    for method in methods:
        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'my_best':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
            normalized = normalized.loc[data.index, :]
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        print("method: {}".format(method))
        print('negative: {}%'.format(
            round(100 * (normalized < 0).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print('< 1000: {}%'.format(
            round(100 * (normalized < 1000).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print()


if __name__ == "__main__":

    save_plots = True
    scenario = 2

    # benchmarks
    plot_benchmarks_cvs_for_methods(scenario=scenario, save_plot=save_plots)
    plot_benchmarks_grouping_coefs_for_methods(scenario=scenario, save_plot=save_plots)
    plot_benchmarks_corrs_for_methods(scenario=scenario, save_plot=save_plots)

    # all samples
    plot_normalized_spectra_for_methods(scenario=scenario, file_ext='png', save_plot=save_plots)
    check_relevant_intensities_for_methods(scenario=scenario)
    plot_samples_corrs_for_methods(scenario=scenario, save_plot=save_plots)  # not very informative

