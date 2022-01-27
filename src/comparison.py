
import numpy, pandas, seaborn, time
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

import ralps, processing
from constants import default_parameters_values
from batch_analysis import plot_batch_cross_correlations, compute_vc_for_samples_types
from batch_analysis import compute_number_of_clusters_with_hdbscan
from batch_analysis import get_sample_cross_correlation_estimate
from utils import combat

# constants
user = 'andreidm'
path = '/Users/{}/ETH/projects/normalization/data/'.format(user)


def plot_benchmarks_corrs_for_methods(scenario=1, save_plot=False):
    """ This method plots heatmaps of batch cross-correlations for benchmark samples. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    if scenario < 3:
        data_path = path + 'filtered_data_v4.csv'
        info_path = path + 'batch_info_v4.csv'
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
        info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'
    else:
        raise ValueError('Indicate scenario.')

    data = ralps.get_data({'data_path': data_path, 'info_path': info_path,
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    batch_info = pandas.read_csv(info_path, keep_default_na=False)

    _, benchmarks = processing.extract_reg_types_and_benchmarks(data)

    for method in methods:

        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'ralps':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0).T

        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T

            # make sure the same sample names are used
            renaming = {}
            for i in range(normalized.shape[0]):
                if normalized.index[i] in data.index:
                    continue
                else:
                    for j in range(data.shape[0]):
                        if normalized.index[i] in data.index[j] and normalized.index[i][-2:] == data.index[j][-2:]:
                            renaming[normalized.index[i]] = data.index[j]
                            break
            # rename using the same prefixes
            normalized = normalized.rename(index=renaming)

        elif method == 'combat':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            normalized = add_prefixes_to_samples_names(normalized, batch_info)

        plot_batch_cross_correlations(normalized, method, {'id': '', 'plots_extension': '.pdf'}, sample_types_of_interest=benchmarks, save_to=save_to, save_plot=save_plot)

    if save_plot:
        print('benchmark correlations saved')


def plot_benchmarks_cvs_for_methods(scenario=1, save_plot=False):
    """ Plots variation coefs for benchmarks. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    all_cvs = pandas.DataFrame()

    if scenario < 3:
        data_path = path + 'filtered_data_v4.csv'
        info_path = path + 'batch_info_v4.csv'
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
        info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'
    else:
        raise ValueError('Indicate scenario.')

    data = ralps.get_data({'data_path': data_path, 'info_path': info_path,
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    batch_info = pandas.read_csv(info_path, keep_default_na=False)

    _, benchmarks = processing.extract_reg_types_and_benchmarks(data)

    for method in methods:

        if method == 'none':
            normalized = data.iloc[:, 1:]

        elif method == 'ralps':
            # hardcode
            normalized = pandas.read_csv(path_to_my_best, index_col=0).T

        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T

            # make sure the same sample names are used
            renaming = {}
            for i in range(normalized.shape[0]):
                if normalized.index[i] in data.index:
                    continue
                else:
                    for j in range(data.shape[0]):
                        if normalized.index[i] in data.index[j] and normalized.index[i][-2:] == data.index[j][-2:]:
                            renaming[normalized.index[i]] = data.index[j]
                            break
            # rename using the same prefixes
            normalized = normalized.rename(index=renaming)
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            normalized = add_prefixes_to_samples_names(normalized, batch_info)

        res = compute_vc_for_samples_types(normalized, benchmarks)
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
    """ Plot statistics for cross-correlations within regularization sample types. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    if scenario < 3:
        data_path = path + 'filtered_data_v4.csv'
        info_path = path + 'batch_info_v4.csv'
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
        info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'
    else:
        raise ValueError('Indicate scenario.')

    data = ralps.get_data({'data_path': data_path, 'info_path': info_path,
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    batch_info = pandas.read_csv(info_path, keep_default_na=False)

    regs, _ = processing.extract_reg_types_and_benchmarks(data)

    corrs = []
    for method in methods:

        if method == 'none':
            normalized = data.iloc[:, 1:]

        elif method == 'ralps':
            normalized = pandas.read_csv(path_to_my_best, index_col=0).T

        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T

            # make sure the same sample names are used
            renaming = {}
            for i in range(normalized.shape[0]):
                if normalized.index[i] in data.index:
                    continue
                else:
                    for j in range(data.shape[0]):
                        if normalized.index[i] in data.index[j] and normalized.index[i][-2:] == data.index[j][-2:]:
                            renaming[normalized.index[i]] = data.index[j]
                            break
            # rename using the same prefixes
            normalized = normalized.rename(index=renaming)

        elif method == 'combat':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            add_prefixes_to_samples_names(normalized, batch_info)

        res = get_sample_cross_correlation_estimate(normalized, regs, percent=25)
        corrs.append(res)

    res = pandas.DataFrame({'method': methods, 'corr': corrs})

    pyplot.figure()
    seaborn.barplot(x='method', y='corr', data=res)
    pyplot.xlabel('Normalization')
    pyplot.ylabel('Correlation sum')
    pyplot.grid()
    pyplot.tick_params(labelrotation=45)
    pyplot.tight_layout()

    if save_plot:
        pyplot.savefig(save_to + 'corrs.pdf')
        print('overall correlations saved')
    else:
        pyplot.show()


def get_grouping_coefs_for_samples(method, clustering, total_clusters, benchmarks):

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
        methods = ['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'normAE', 'ralps']
        path_to_my_best = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/best_model/2d48bfb2_best/normalized_2d48bfb2.csv'
        path_to_others = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/other_methods/'
        save_to = '/Users/andreidm/ETH/projects/normalization/res/no_reference_samples/other_methods/plots/'
    elif scenario == 2:
        methods = ['none', 'normAE', 'ralps']
        path_to_my_best = path_to_my_best_method_2
        path_to_others = path_to_other_methods_2
        save_to = '/Users/andreidm/ETH/projects/normalization/res/fake_reference_samples/other_methods/plots/'
    elif scenario == 3:
        # it's actually scenario 2, but on another dataset
        methods = ['none', 'lev+eig', 'pqn+pow', 'combat', 'eigenMS', 'waveICA', 'normAE', 'ralps']
        path_to_my_best = '/Users/andreidm/ETH/projects/normalization/res/sarahs/b2a75470/normalized_b2a75470.csv'
        path_to_others = '/Users/andreidm/ETH/projects/normalization/res/sarahs/other_methods/'
        save_to = '/Users/andreidm/ETH/projects/normalization/res/sarahs/other_methods/plots/'
    else:
        raise ValueError("Indicate application scenario.")

    return methods, path_to_my_best, path_to_others, save_to


def plot_benchmarks_grouping_coefs_for_methods(scenario=1, save_plot=False):
    """ Plot grouping coefs for benchmark samples after normalization. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    if scenario < 3:
        data_path = path + 'filtered_data_v4.csv'
        info_path = path + 'batch_info_v4.csv'
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
        info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'
    else:
        raise ValueError('Indicate scenario.')

    data = ralps.get_data({'data_path': data_path, 'info_path': info_path,
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    batch_info = pandas.read_csv(info_path, keep_default_na=False)

    _, benchmarks = processing.extract_reg_types_and_benchmarks(data)

    pars = {'latent_dim': data.shape[1]-1,
            'n_batches': len(data['batch'].unique()),
            'n_replicates': default_parameters_values['n_replicates']}

    all_grouping_coefs = pandas.DataFrame()

    for method in methods:

        if method == 'none':
            normalized = data
        elif method == 'ralps':
            normalized = pandas.read_csv(path_to_my_best, index_col=0).T
            # keep the ordering as in initial data to insert batches
            normalized = normalized.loc[data.index, :]
            normalized['batch'] = data['batch']

        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T

            # make sure the same sample names are used
            renaming = {}
            for i in range(normalized.shape[0]):
                if normalized.index[i] in data.index:
                    continue
                else:
                    for j in range(data.shape[0]):
                        if normalized.index[i] in data.index[j] and normalized.index[i][-2:] == data.index[j][-2:]:
                            renaming[normalized.index[i]] = data.index[j]
                            break
            # rename using the same prefixes
            normalized = normalized.rename(index=renaming)

            # keep the ordering as in initial data to insert batches
            normalized = normalized.loc[data.index, :]
            normalized['batch'] = data['batch']

        elif method == 'combat':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            # keep the ordering as in initial data to insert batches
            normalized = normalized.loc[data.index, :]
            normalized['batch'] = data['batch']

        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            normalized = add_prefixes_to_samples_names(normalized, batch_info)
            # keep the ordering as in initial data to insert batches
            normalized = normalized.loc[data.index, :]
            normalized['batch'] = data['batch']

        clustering, total_clusters = compute_number_of_clusters_with_hdbscan(normalized, pars, benchmarks, print_info=False)
        grouping_dict = get_grouping_coefs_for_samples(method, clustering, total_clusters, benchmarks)

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
    """ This methods plots spectra of normalized data. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    if scenario < 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/filtered_data_with_mz.csv'
        # hardcode colors for batches from our dataset
        batches = ['0108', '0110', '0124', '0219', '0221', '0304', '0306']
        color_dict = dict(zip(batches, 'kgrcmyb'))
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/data_with_mzs.csv'
        # hardcode colors for batches from Sarah's dataset
        batches = ['Batch1', 'Batch2', 'Batch3', 'Batch4', 'Batch5', 'Batch6', 'Batch7']
        color_dict = dict(zip(batches, 'kgrcmyb'))
    else:
        raise ValueError('Indicate scenario.')

    data_with_mz = pandas.read_csv(data_path, index_col=0)
    mz = data_with_mz['mz'].values
    data_with_mz = data_with_mz.iloc[:, 1:].T

    for method in methods:

        if method == 'none':
            normalized = data_with_mz

        elif method == 'ralps':
            normalized = pandas.read_csv(path_to_my_best, index_col=0)
            if scenario == 3:
                # previously the output was transposed
                normalized = normalized.T

        elif method == 'normAE':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T

            # make sure the same sample names are used
            renaming = {}
            for i in range(normalized.shape[0]):
                if normalized.index[i] in data_with_mz.index:
                    continue
                else:
                    for j in range(data.shape[0]):
                        if normalized.index[i] in data_with_mz.index[j] and normalized.index[i][-2:] == data_with_mz.index[j][-2:]:
                            renaming[normalized.index[i]] = data_with_mz.index[j]
                            break
            # rename using the same prefixes
            normalized = normalized.rename(index=renaming)

            normalized = normalized.loc[data_with_mz.index, :]  # keep the ordering

        elif method == 'combat':
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)
            normalized = normalized.loc[data_with_mz.index, :]  # keep the ordering

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


def add_prefixes_to_samples_names(data, batch_info):

    # create prefixes for grouping
    new_index = data.index.values
    groups_indices = numpy.where(numpy.isin(batch_info['group'].astype('str'), ('0', ''), invert=True))[0]
    new_index[groups_indices] = 'group_' + batch_info['group'][groups_indices].astype('str') + '_' + new_index[groups_indices]

    # create prefixes for benchmarks
    benchmarks_indices = numpy.where(numpy.isin(batch_info['benchmark'].astype('str'), ('0', ''), invert=True))[0]
    new_index[benchmarks_indices] = 'bench_' + batch_info['benchmark'][benchmarks_indices].astype('str') + '_' + new_index[benchmarks_indices]
    data.index = new_index

    return data


def check_relevant_intensities_for_methods(scenario=1):
    """ Methods 'lev+eig', 'pqn+pow', 'normAE' are excluded by default, since they don't output intensities. """

    methods, path_to_my_best, path_to_others, save_to = get_paths_and_methods(scenario)

    if scenario < 3:
        data_path = path + 'filtered_data_v4.csv'
        info_path = path + 'batch_info_v4.csv'
    elif scenario == 3:
        data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
        info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'
    else:
        raise ValueError('Indicate scenario.')

    data = ralps.get_data({'data_path': data_path, 'info_path': info_path,
                            'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})

    for method in methods:
        if method == 'none':
            normalized = data.iloc[:, 1:]
        elif method == 'ralps':
            normalized = pandas.read_csv(path_to_my_best, index_col=0).T
        # elif method == 'normAE':
        #     # it doesn't output intensities, but I'm curious how much negative values they produce
        #     normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0).T
        elif method in ['lev+eig', 'pqn+pow', 'normAE']:
            continue
        else:
            normalized = pandas.read_csv(path_to_others + '{}.csv'.format(method), index_col=0)

        print("method: {}".format(method))
        print('min: {}'.format(int(normalized.min().min())))
        print('max: {}'.format(int(normalized.max().max())))
        print('negative: {}%'.format(round(100 * (normalized < 0).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print('< 1000: {}%'.format(round(100 * (normalized < 1000).sum().sum() / normalized.shape[0] / normalized.shape[1], 1)))
        print()


def plot_percent_of_unique_values(save_to='/Users/andreidm/ETH/projects/normalization/res/sarahs/other_methods/plots/'):
    """ This is only to compare with NormAE and show that their solution is actually collapsed. """

    data_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/filtered_data.csv'
    info_path = '/Users/andreidm/ETH/projects/normalization/data/sarah/batch_info.csv'

    original_data = ralps.get_data({'data_path': data_path, 'info_path': info_path, 'min_relevant_intensity': default_parameters_values['min_relevant_intensity']})
    original_data = original_data.iloc[:, 1:].T
    reg_samples_cols = [x for x in original_data.columns if 'MDAMB231_Medium1_BREAST_JB' in x]
    original_values = original_data.loc[:, reg_samples_cols].values.flatten()
    print('original:', len(set(original_values)) / len(original_values) * 100)

    normae_path = '/Users/andreidm/ETH/projects/normalization/res/sarahs/other_methods/normAE.csv'
    normae_normalized = pandas.read_csv(normae_path, index_col=0)
    reg_samples_cols = [x for x in normae_normalized.columns if 'MDAMB231_Medium1_BREAST_JB' in x]
    normae_values = normae_normalized.loc[:, reg_samples_cols].values.flatten()
    print('normae:', len(set(normae_values)) / len(normae_values) * 100)

    ralps_path = '/Users/andreidm/ETH/projects/normalization/res/sarahs/b2a75470/normalized_b2a75470.csv'
    ralps_normalized = pandas.read_csv(ralps_path, index_col=0)
    reg_samples_cols = [x for x in ralps_normalized.columns if 'MDAMB231_Medium1_BREAST_JB' in x]
    ralps_values = ralps_normalized.loc[:, reg_samples_cols].values.flatten()
    print('ralps:', len(set(ralps_values)) / len(ralps_values) * 100)

    data = pandas.DataFrame({
        'method': ['None', 'ralps', 'NormAE'],
        'percent': [len(set(original_values)) / len(original_values) * 100,
                    len(set(ralps_values)) / len(ralps_values) * 100,
                    len(set(normae_values)) / len(normae_values) * 100]
    })

    seaborn.set_theme(style="whitegrid")
    seaborn.barplot(x="method", y="percent", data=data)
    pyplot.title('MDAMB231_Medium1_BREAST_JB: percent of unique values')
    pyplot.savefig(save_to + 'unique_values.pdf')


def compute_percent_of_increased_vcs_for_methods(path_to_init_data, path_to_other_methods, allowed_percent=0.05):
    """ This method computes percent of increased (compared to initial data) VCs for samples. """

    initial_data = pandas.read_csv(path_to_init_data, index_col=0).T

    percent_of_increased_vc = {}
    for method in ['normAE', 'combat', 'eigenMS', 'lev+eig', 'pqn+pow', 'waveICA']:

        if method == 'normAE':
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0).T
        else:
            normalized = pandas.read_csv(path_to_other_methods + '{}.csv'.format(method), index_col=0)

        count = 0
        for sample in initial_data.index:
            init_vc = initial_data.loc[sample].std() / initial_data.loc[sample].mean()
            sample_vc = normalized.loc[sample].std() / normalized.loc[sample].mean()
            if sample_vc - init_vc > init_vc * allowed_percent:
                count += 1
        percent_of_increased_vc[method] = int(count / normalized.shape[0] * 100)

    for key, value in percent_of_increased_vc.items():
        print('{}: {}%'.format(key, value))


def plot_single_spectrum(mz, data, title):
    """ Plots a spectrum of data for the benchmarking dataset. """

    # hardcoded batch ids and colors
    batches = ['0108', '0110', '0124', '0219', '0221', '0304', '0306']
    color_dict = dict(zip(batches, 'kgrcmyb'))

    samples = data.index
    pyplot.figure(figsize=(8, 6))
    for i in range(data.shape[0]):
        batch_id = [batch in samples[i] for batch in batches].index(True)
        color = color_dict[batches[batch_id]]
        if color == 'b':
            pyplot.plot(mz, data.values[i, :], '{}o'.format(color), alpha=0.2)
        else:
            pyplot.plot(mz, data.values[i, :], '{}o'.format(color), alpha=0.4)
    pyplot.title(title)
    pyplot.xlabel('mz')
    pyplot.ylabel('intensity')
    pyplot.grid()
    pyplot.show()


if __name__ == "__main__":

    # save_plots = False
    # scenario = 3
    #
    # # benchmarks
    # plot_benchmarks_cvs_for_methods(scenario=scenario, save_plot=save_plots)
    # plot_benchmarks_grouping_coefs_for_methods(scenario=scenario, save_plot=save_plots)
    # plot_benchmarks_corrs_for_methods(scenario=scenario, save_plot=save_plots)
    #
    # # all samples
    # check_relevant_intensities_for_methods(scenario=scenario)
    # plot_samples_corrs_for_methods(scenario=scenario, save_plot=save_plots)
    # plot_normalized_spectra_for_methods(scenario=scenario, file_ext='png', save_plot=save_plots)
    #
    # plot_percent_of_unique_values()

    compute_percent_of_increased_vcs_for_methods(
        'D:\ETH\projects\\normalization\data\\filtered_data.csv',
        'D:\ETH\projects\\normalization\\res\\all_refs_other_methods\\'
    )

    original_data = pandas.read_csv('D:\ETH\projects\\normalization\data\\filtered_data_with_mz.csv', index_col=0)
    mz = original_data['mz']

    original_data = original_data.drop(columns=['name', 'mz']).T
    plot_single_spectrum(mz, original_data, 'Original')

    normalized_1 = pandas.read_csv('D:\ETH\projects\\normalization\\res\\SRM+SPP\\dc98d5fc\\normalized_dc98d5fc.csv', index_col=0).T
    plot_single_spectrum(mz, normalized_1, 'dc98d5fc, rec_loss > 4')