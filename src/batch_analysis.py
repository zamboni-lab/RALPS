
import numpy, pandas, seaborn, umap, time, hdbscan, torch, matplotlib
from matplotlib import pyplot
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA


def compute_samples_vcs(data):
    """ This method computes sample-wise variation coefs. """
    vcs = []
    for i in range(data.shape[0]):
        vcs.append(data.iloc[i,:].std() / data.iloc[i,:].mean())
    return vcs


def compute_percent_of_increased_vcs(normalized, init_vcs, increase_percent=0.05):
    """ This method computes percent of increased VCs in normalized data, compared to the initial ones.
        Note that normalized and initial data have the same index. """
    count = 0
    for i in range(normalized.shape[0]):
        norm_vc = normalized.iloc[i,:].std() / normalized.iloc[i,:].mean()
        if norm_vc - init_vcs[i] > init_vcs[i] * increase_percent:
            count += 1
    return round(count / normalized.shape[0] * 100, 1)


def get_samples_by_types_dict(samples_names, types_of_interest):
    """ Create a dict like this: {'sample_type_1': ['sample_1', ..., 'sample_1_diluted'], 'sample_type_2': [...] }  """

    samples_by_types = {}
    for i, sample in enumerate(samples_names):
        # check which type this sample has
        for type in types_of_interest:
            if type in sample and type not in samples_by_types:
                # if new type, put in the dict, create a list
                samples_by_types[type] = [sample]
            elif type in sample and type in samples_by_types:
                # if type already exists in the dict, append sample
                samples_by_types[type].append(sample)
            else:
                pass

    return samples_by_types


def get_shortened_samples_names(samples_names):
    """ This method shortens samples' names for better visualization. """

    for i in [-5, -4, -3, -2]:
        new_names = ['_'.join(name.split('_')[-i:]) for name in samples_names]
        are_all_short = sum([len(name) > 10 for name in new_names]) == 0
        if are_all_short:
            return new_names
    return [name[-10:] for name in samples_names]


def plot_batch_cross_correlations(data, method_name, parameters, sample_types_of_interest, save_to='/Users/andreidm/ETH/projects/normalization/res/', save_plot=False):
    """ This method plots heatmaps of intra-batch correaltions of the same samples of interest. """

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    # plot one by one
    for i, s_type in enumerate(samples_by_types):
        pyplot.figure()
        df = data.loc[numpy.array(samples_by_types[s_type]), :]
        df = df.T  # transpose to call corr() on samples, not metabolites
        df = df.reindex(sorted(df.columns), axis=1)  # sort column names
        df.columns = get_shortened_samples_names(df.columns)
        df = df.corr()

        seaborn.heatmap(df, vmin=0, vmax=1)
        pyplot.title(s_type)
        pyplot.suptitle('Cross correlations: {}'.format(method_name))
        pyplot.tight_layout()

        if save_plot:
            pyplot.savefig(save_to / 'correlations_{}_{}_{}.{}'.format(s_type, method_name.replace(' ', '_'), parameters['id'], parameters['plots_extension']))
        else:
            pyplot.show()
        pyplot.close()


def get_sample_cross_correlation_estimate(data, sample_types_of_interest, percent=50):
    """ This method computes mean intra-batch correlations for the samples of interest.
        A simple statistic is computed then to give an estimate (ex., median, or 25th percentile). """

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    corrs = []
    for i, type in enumerate(samples_by_types):
        df = data.loc[numpy.array(samples_by_types[type]), :]
        df = df.T.corr()  # transpose to call corr() on samples, not metabolites
        values = df.values.flatten()
        corrs.append(numpy.percentile(values, percent))

    return numpy.mean(corrs)


def compute_vc_for_batches(data, batch_labels):
    """ This method computes variation coefs for entire batches. """

    batch_vcs = {}
    for label in batch_labels.unique():
        batch_values = data.loc[batch_labels == label, :].values.flatten()
        batch_vcs[label] = numpy.std(batch_values) / numpy.mean(batch_values)

    return batch_vcs


def compute_vc_for_samples_types(data, sample_types_of_interest):
    """ This method computes variation coefs for samples of interest. """

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    vc_dict = {}
    for i, type in enumerate(samples_by_types):
        values = data.loc[numpy.array(samples_by_types[type]), :].values
        values = values[values > 0]  # exclude negative values, that the model might predict
        values = values.flatten()
        vc_dict[type] = numpy.std(values) / numpy.mean(values)

    return vc_dict


def get_pca_reduced_data(data, parameters):

    transformer = PCA(n_components=parameters['latent_dim'])
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    pca_reduced = pandas.DataFrame(transformer.fit_transform(scaled_data))

    return pca_reduced


def plot_batch_vcs(vc_batch_initial, vc_batch_normalized, parameters, save_to=None):
    """ This methos plots variation coefs of batches before and after normalization. """

    data = {'batch': [], 'vc': [], 'label': []}
    # merge into a single container
    for key, value in vc_batch_initial.items():
        data['batch'].append(key)
        data['vc'].append(value)
        data['label'].append('Initial')
    for key, value in vc_batch_normalized.items():
        data['batch'].append(key)
        data['vc'].append(value)
        data['label'].append('Normalized')

    data = pandas.DataFrame(data)
    data = data.sort_values('batch')

    pyplot.figure()
    seaborn.set_theme(style="whitegrid")
    seaborn.barplot(x='batch', y='vc', hue='label', data=data)
    pyplot.title('Batch variation coefs')
    pyplot.legend(bbox_to_anchor=(1.01, 1))
    pyplot.tight_layout()
    if save_to:
        pyplot.savefig(save_to / 'batch_vc_initial_vs_normalized_{}.{}'.format(parameters['id'], parameters['plots_extension']))
        pyplot.close()
    else:
        pyplot.show()


def plot_full_data_umaps(data, reconstruction, batch_labels, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):
    """ This method plots UMAP embeddings of PCA reduced data (initial, normalized and model encoded). """

    # plot initial data
    initial_reduced = get_pca_reduced_data(data, parameters)
    normalized_reduced = get_pca_reduced_data(reconstruction, parameters)

    plot_umap(initial_reduced, batch_labels.values, parameters, 'initial data', save_to=save_to)
    plot_umap(normalized_reduced, batch_labels.values, parameters, 'normalized data', save_to=save_to)


def plot_umap(data, batch_labels, parameters, plot_name, metric='braycurtis', save_to='/Users/andreidm/ETH/projects/normalization/res/'):
    """ This method visualizes UMAP embeddings of the data encodings.
        It helps to assess batch effects on the high level."""

    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])

    reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=0.9, random_state=77)
    embeddings = reducer.fit_transform(data)

    # plot coloring batches
    seaborn.set()
    pyplot.figure(figsize=(7, 6))
    palette = seaborn.color_palette('deep', n_colors=len(set(batch_labels)))

    seaborn.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=batch_labels, alpha=.8, palette=palette)
    pyplot.legend(title='Batch', loc=1, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}'.format(plot_name, neighbors, metric))
    pyplot.tight_layout()
    pyplot.savefig(save_to / 'umap_{}_{}.{}'.format(plot_name.replace(' ', '_'), parameters['id'], parameters['plots_extension']))
    pyplot.close()


def plot_full_data_umap_with_benchmarks(encodings, method_name, parameters, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):
    """ Produces a plot with UMAP embeddings, colored after specified samples.
        Seems to be not very useful. """

    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])
    metric = 'braycurtis'

    reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=0.9, random_state=77)
    embeddings = reducer.fit_transform(encodings.values)

    # define colors of benchmark samples
    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    samples_colors = []
    for sample in encodings.index.values:

        for i, s_type in enumerate(samples_by_types):
            if sample in samples_by_types[s_type]:
                samples_colors.append(s_type)
                break
            elif i == len(samples_by_types) - 1:
                samples_colors.append('All the rest')
            else:
                pass

    # arrange colors
    data = pandas.DataFrame({'xs': embeddings[:, 0], 'ys': embeddings[:, 1], 'color': samples_colors})
    data = data.sort_values(by='color')

    # plot coloring benchmark samples
    seaborn.set()
    pyplot.figure(figsize=(7, 6))
    paired_palette = list(seaborn.color_palette('Paired'))
    my_palette = [paired_palette[i] for i in range(len(paired_palette)) if i in [0, 1, 3, 5, 7, 9]]
    my_palette = seaborn.set_palette(seaborn.color_palette(my_palette))
    ax = seaborn.scatterplot(x='xs', y='ys', hue='color', alpha=0.9, palette=my_palette, data=data)
    h, l = ax.get_legend_handles_labels()
    pyplot.legend(h[1:], l[1:], title='Samples', loc=1, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}: n={}, metric={}'.format(method_name, neighbors, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_{}_{}.pdf'.format(method_name.replace(' ', '_'), parameters['id']))
    pyplot.close()


def compute_number_of_clusters_with_hdbscan(encodings, parameters, sample_types_of_interest, print_info=True):
    """ This method applied HDBSCAN clustering on the encodings,
        and returns assigned clusters to the samples of interest.  """

    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    n_comp = int(parameters['latent_dim'] / 3)
    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])
    metric = 'braycurtis'

    reducer = umap.UMAP(n_components=n_comp, n_neighbors=neighbors, metric=metric, min_dist=0.1, random_state=77)
    embeddings = reducer.fit_transform(values)

    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=neighbors, allow_single_cluster=False)
    clusterer.fit(embeddings)

    total = clusterer.labels_.max() + 1
    if print_info:
        print('CLUSTERING INFO:\n')
        print('Total of clusters:', total)

    labels_dict = {}
    for i, type in enumerate(samples_by_types):
        indices_of_type = [list(encodings.index.values).index(x) for x in samples_by_types[type]]
        type_batches = batches[numpy.array(indices_of_type)]

        type_labels = clusterer.labels_[numpy.array(indices_of_type)]
        type_probs = clusterer.probabilities_[numpy.array(indices_of_type)]
        type_outlier_probs = clusterer.outlier_scores_[numpy.array(indices_of_type)]

        labels_dict[type] = list(type_labels)

        if print_info:
            print('{}:'.format(type))
            print('n clusters:', len(set(type_labels)))
            print('true batches:', list(type_batches))
            print('labels:', list(type_labels))
            print('probs:', list(type_probs))
            print('outlier probs:', list(type_outlier_probs), '\n')

    return labels_dict, total


if __name__ == '__main__':

    # data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data.csv')
    # data = data.iloc[:, 1:]
    #
    # pars = {'id': '', 'plots_extension': 'pdf'}
    # plot_batch_cross_correlations(data.T, 'initial samples', '', ['P1_FA_0001', 'P2_SF_0001',
    #                                                                'P2_SFA_0001', 'P2_SRM_0001',
    #                                                                'P2_SFA_0002', 'P1_FA_0008'])
    #
    # res = compute_cv_for_samples_types(data.T, ['P1_FA_0001', 'P2_SF_0001',
    #                                             'P2_SFA_0001', 'P2_SRM_0001',
    #                                             'P2_SFA_0002', 'P1_FA_0008'])
    # print(res)
    #
    # # get encodings of the SEPARATELY TRAINED autoencoder
    # encodings = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/autoencoder/encodings.csv', index_col=0)
    #
    # pars = {'n_batches': 7, 'n_replicates': 3, 'id': '', 'plots_extension': 'pdf'}
    # plot_encodings_umap(encodings, 'initial samples', pars,
    #                     save_to='/Users/andreidm/ETH/projects/normalization/res/')
    #
    # pars = {'latent_dim': 100, 'n_batches': 7, 'n_replicates': 3}
    # res, _ = compute_number_of_clusters_with_hdbscan(encodings, pars, ['P1_FA_0001', 'P2_SF_0001',
    #                                                                    'P2_SFA_0001', 'P2_SRM_0001',
    #                                                                    'P2_SFA_0002', 'P1_FA_0008'],
    #                                                  print_info=True)
    # print(res)

    # encodings_raw = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/SRM+SPP-disc/c461e5f6/encodings_c461e5f6.csv', index_col=0)
    # encodings_normalized = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/P2_SRM_0001+P2_SPP_0001/da9e81db/encodings_da9e81db.csv', index_col=0)
    # all_samples = encodings_raw.index

    from ralps import get_data
    raw_data = get_data({'data_path': '/Users/andreidm/ETH/projects/normalization/data/filtered_data_v5.csv',
                               'info_path': '/Users/andreidm/ETH/projects/normalization/data/batch_info_v5_SRM+SPP.csv',
                               'min_relevant_intensity': 1000})

    transformer = PCA()
    scaler = StandardScaler()

    scaled_raw_data = scaler.fit_transform(raw_data.iloc[:, 1:])
    encodings_raw = transformer.fit_transform(scaled_raw_data)
    encodings_raw = pandas.DataFrame(encodings_raw)
    encodings_raw.insert(0, 'batch', raw_data.iloc[:, 0].values)
    encodings_raw.index = raw_data.index
    encodings_raw.columns = raw_data.columns

    normalized_data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/SRM+SPP-disc/c461e5f6/normalized_c461e5f6.csv', index_col=0).T
    scaled_normalized_data = scaler.fit_transform(normalized_data)
    encodings_normalized = transformer.fit_transform(scaled_normalized_data)
    encodings_normalized = pandas.DataFrame(encodings_normalized)
    encodings_normalized.index = normalized_data.index
    encodings_normalized.columns = normalized_data.columns
    batch = []
    for sample in normalized_data.index:
        if '_0108_' in sample:
            batch.append(1)
        elif '_0110_' in sample:
            batch.append(2)
        elif '_0124_' in sample:
            batch.append(3)
        elif '_0219_' in sample:
            batch.append(4)
        elif '_0221_' in sample:
            batch.append(5)
        elif '_0304_' in sample:
            batch.append(6)
        else:
            batch.append(7)

    save_to = '/Users/andreidm/ETH/projects/normalization/res/SRM+SPP-disc/'

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'before', 'plots_extension': 'pdf'}
    plot_umap(encodings_raw, numpy.array(batch), pars, 'initial', save_to=save_to)
    plot_umap(encodings_normalized, numpy.array(batch), pars, 'normalized', save_to=save_to)

    samples = ['P2_SRM_0001', 'P2_SPP_0001']

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'reg_samples_before'}
    plot_full_data_umap_with_benchmarks(encodings_raw, 'initial', pars, sample_types_of_interest=samples, save_to=save_to)
    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'reg_samples_after'}
    plot_full_data_umap_with_benchmarks(encodings_normalized, 'normalized', pars, sample_types_of_interest=samples, save_to=save_to)

    samples = ['P1_FA_0008', 'P2_SF_0001', 'P2_SFA_0002', 'P1_FA_0001', 'P2_SFA_0001']

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'benchmarks_before'}
    plot_full_data_umap_with_benchmarks(encodings_raw, 'initial', pars, sample_types_of_interest=samples, save_to=save_to)
    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'benchmarks_after'}
    plot_full_data_umap_with_benchmarks(encodings_normalized, 'normalized', pars, sample_types_of_interest=samples, save_to=save_to)







