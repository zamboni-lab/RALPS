
import numpy, pandas, seaborn, umap, time
from matplotlib import pyplot
import hdbscan, torch
from sklearn.preprocessing import RobustScaler

from src.constants import user, path_to_my_best_method_1
from src.models.ae import Autoencoder
from src.constants import benchmark_sample_types as benchmarks
from src.constants import shared_perturbations as all_samples


def get_samples_by_types_dict(samples_names, types_of_interest):
    """ Create a dict like this: {'P1_FA_0001': ['P1_FA_0001_0306_0', ..., 'P1_FA_0001_0306_2'], ...}  """

    samples_by_types = {}
    for i, sample in enumerate(samples_names):
        # check which type this sample has
        for type in types_of_interest:
            if type in sample and type not in samples_by_types:
                # if new type, putt in the dict, create a list
                samples_by_types[type] = [sample]
                break
            elif type in sample and type in samples_by_types:
                # if type already exists in the dict, append sample
                samples_by_types[type].append(sample)
                break
            else:
                pass

    return samples_by_types


def plot_batch_cross_correlations(data, method_name, id, sample_types_of_interest=None, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user), save_plot=False):

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    if sample_types_of_interest is None:
        for i, type in enumerate(samples_by_types):
            df = data.loc[numpy.array(samples_by_types[type]), :]
            df = df.T  # transpose to call corr() on samples, not metabolites
            df.columns = [x[-6:] for x in df.columns]
            df = df.corr()

            seaborn.heatmap(df, vmin=0, vmax=1)
            pyplot.title('Cross correlations: {}: {}'.format(type, method_name))
            pyplot.tight_layout()
            pyplot.show()

    else:
        pyplot.figure(figsize=(12,8))

        for i, type in enumerate(samples_by_types):
            df = data.loc[numpy.array(samples_by_types[type]), :]
            df = df.T  # transpose to call corr() on samples, not metabolites
            df.columns = sorted([x[-6:] for x in df.columns])
            df = df.corr()

            ax = pyplot.subplot(2, 3, i+1)
            seaborn.heatmap(df, vmin=0, vmax=1)
            ax.set_title(type)

        pyplot.suptitle('Cross correlations: {}'.format(method_name))
        pyplot.tight_layout()

        if save_plot:
            pyplot.savefig(save_to + 'correlations_{}_{}.pdf'.format(method_name.replace(' ', '_'), id))
        else:
            pyplot.show()


def get_sample_cross_correlation_estimate(data, percent=50, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    corrs = []
    for i, type in enumerate(samples_by_types):
        df = data.loc[numpy.array(samples_by_types[type]), :]
        df = df.T.corr()  # transpose to call corr() on samples, not metabolites
        values = df.values.flatten()
        corrs.append(numpy.percentile(values, percent))

    return numpy.sum(corrs)


def compute_cv_for_samples_types(data, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    cv_dict = {}
    for i, type in enumerate(samples_by_types):
        values = data.loc[numpy.array(samples_by_types[type]), :].values
        values = values[values > 0]  # exclude negative values, that the model might predict
        values = values.flatten()
        cv_dict[type] = numpy.std(values) / numpy.mean(values)

    return cv_dict


def plot_full_dataset_umap(encodings, method_name, parameters, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])
    metric = 'braycurtis'

    reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=0.9, random_state=77)
    embeddings = reducer.fit_transform(values)

    # plot coloring batches
    seaborn.set()
    pyplot.figure(figsize=(8, 6))

    seaborn.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=batches, alpha=.8, palette=seaborn.color_palette('deep', n_colors=len(set(batches))))
    pyplot.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}: n={}, metric={}'.format(method_name, neighbors, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_batches_{}_{}.pdf'.format(method_name.replace(' ', '_'), parameters['id']))


def plot_full_data_umap_with_benchmarks(encodings, method_name, parameters, sample_types_of_interest=None, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):
    """ Produces a plot with UMAP embeddings, colored after specified samples.
        Seems to be not very useful... """

    values = encodings.iloc[:, 1:].values

    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])
    metric = 'braycurtis'

    reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=0.9, random_state=77)
    embeddings = reducer.fit_transform(values)

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
    pyplot.figure(figsize=(8, 6))
    ax = seaborn.scatterplot(x='xs', y='ys', hue='color', alpha=.9, palette=seaborn.color_palette('Paired', n_colors=len(set(samples_colors))), data=data)
    h, l = ax.get_legend_handles_labels()
    pyplot.legend(h[1:], l[1:], title='Samples', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}: n={}, metric={}'.format(method_name, neighbors, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_benchmarks_{}_{}.pdf'.format(method_name.replace(' ', '_'), parameters['id']))


def compute_number_of_clusters_with_hdbscan(encodings, parameters, print_info=True, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    n_comp = int(parameters['latent_dim'] / 3)
    neighbors = int(parameters['n_batches'] * parameters['n_replicates'])
    metric = 'braycurtis'

    reducer = umap.UMAP(n_components=n_comp, n_neighbors=neighbors, metric=metric, min_dist=0.1, random_state=77)
    embeddings = reducer.fit_transform(values)

    numpy.random.seed(77)  # set seed to make hdbscan results comparable across training epochs
    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=neighbors, allow_single_cluster=False)
    clusterer.fit(embeddings)
    numpy.random.seed(int(1000 * time.time()) % 2**32)  # cancel seed, as it might affect later randomization

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

    data = pandas.read_csv('/Users/{}/ETH/projects/normalization/data/filtered_data.csv'.format(user))
    data = data.iloc[:, 3:]

    plot_batch_cross_correlations(data.T, 'original samples', '',
                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                            'P2_SFA_0002', 'P1_FA_0008'])

    res = compute_cv_for_samples_types(data.T, sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                       'P2_SFA_0001', 'P2_SRM_0001',
                                                                       'P2_SFA_0002', 'P1_FA_0008'])
    print(res)

    # get encodings of the SEPARATELY TRAINED autoencoder
    encodings = pandas.read_csv('/Users/{}/ETH/projects/normalization/res/encodings.csv'.format(user), index_col=0)

    pars = {'n_batches': 7, 'n_replicates': 3, 'id': ''}
    plot_full_dataset_umap(encodings, 'original samples', pars,
                           save_to='/Users/andreidm/ETH/projects/normalization/res/')

    pars = {'latent_dim': 100, 'n_batches': 7, 'n_replicates': 3}
    res, _ = compute_number_of_clusters_with_hdbscan(encodings, pars, print_info=True,
                                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                                            'P2_SFA_0002', 'P1_FA_0008'])
    print(res)

    encodings_raw = pandas.read_csv('/Users/{}/ETH/projects/normalization/res/autoencoder/encodings.csv'.format(user), index_col=0)
    encodings_normalized = pandas.read_csv('/Users/{}/ETH/projects/normalization/res/best_model/2d48bfb2/encodings_2d48bfb2.csv'.format(user), index_col=0)

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'ae'}
    plot_full_dataset_umap(encodings_raw, 'original', pars)
    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': '2d48bfb2'}
    plot_full_dataset_umap(encodings_normalized, 'normalized', pars)

    samples = [sample for sample in all_samples if 'SRM_000' in sample]

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'ae_SRM'}
    plot_full_data_umap_with_benchmarks(encodings_raw, 'original', pars, sample_types_of_interest=samples)
    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': '2d48bfb2_SRM'}
    plot_full_data_umap_with_benchmarks(encodings_normalized, 'normalized', pars, sample_types_of_interest=samples)

    samples = [sample for sample in all_samples if '_SPP_000' in sample]

    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': 'ae_SPP'}
    plot_full_data_umap_with_benchmarks(encodings_raw, 'original', pars, sample_types_of_interest=samples)
    pars = {'n_features': 170, 'latent_dim': 50, 'n_batches': 7, 'n_replicates': 3, 'id': '2d48bfb2_SPP'}
    plot_full_data_umap_with_benchmarks(encodings_normalized, 'normalized', pars, sample_types_of_interest=samples)







