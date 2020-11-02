
import numpy, pandas, seaborn, umap
from matplotlib import pyplot
import hdbscan


def get_samples_by_types_dict(samples_names, types_of_interest):
    """ Create a dict like this: {'P1_FA_0001': ['P1_FA_0001_0306_0', ..., 'P1_FA_0001_0306_2'], ...}  """

    if types_of_interest is None:
        # get unique types + filter out diluted samples
        types_of_interest = list(set(['_'.join(x.split('_')[:3]) for x in samples_names if x.split('_')[2] == '0001']))

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


def plot_batch_cross_correlations(data, method_name, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

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
        # pyplot.show()
        pyplot.savefig(save_to + 'correlations_{}.pdf'.format(method_name.replace(' ', '_')))


def compute_cv_for_samples_types(data, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(data.index.values, sample_types_of_interest)

    cv_dict = {}
    for i, type in enumerate(samples_by_types):
        values = data.loc[numpy.array(samples_by_types[type]), :].values
        values = values.flatten()
        cv_dict[type] = numpy.std(values) / numpy.mean(values)

    return cv_dict


def plot_full_dataset_umap(encodings, method_name, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    random_seed = 666
    metric = 'braycurtis'
    n = 20

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=0.9, random_state=random_seed)
    embeddings = reducer.fit_transform(values)

    # plot coloring batches
    seaborn.set()
    pyplot.figure(figsize=(8, 6))

    seaborn.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=batches, alpha=.8, palette=seaborn.color_palette('deep', n_colors=len(set(batches))))
    pyplot.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}: n={}, metric={}'.format(method_name, n, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_batches_{}.pdf'.format(method_name.replace(' ', '_')))

    # define colors of benchmark samples
    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    samples_colors = []
    for sample in encodings.index.values:

        for i, type in enumerate(samples_by_types):
            if sample in samples_by_types[type]:
                samples_colors.append(type)
                break
            elif i == len(samples_by_types) - 1:
                samples_colors.append('Other')
            else:
                pass

    # plot coloring benchmark samples
    seaborn.set()
    pyplot.figure(figsize=(8, 6))

    seaborn.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=samples_colors, alpha=.9, palette=seaborn.color_palette('Paired', n_colors=len(set(samples_colors))))
    pyplot.legend(title='Samples', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    pyplot.title('UMAP: {}: n={}, metric={}'.format(method_name, n, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_benchmarks_{}.pdf'.format(method_name.replace(' ', '_')))


def compute_number_of_clusters_with_hdbscan(encodings, print_info=True, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    random_seed = 831
    metric = 'braycurtis'
    n = 20

    reducer = umap.UMAP(n_components=30, n_neighbors=n, metric=metric, min_dist=0.1, random_state=random_seed)
    embeddings = reducer.fit_transform(values)

    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=n, allow_single_cluster=False)
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

    data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data.csv')
    data = data.iloc[:, 3:]

    # plot_batch_cross_correlations(data.T, 'original samples',
    #                               sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                         'P2_SFA_0001', 'P2_SRM_0001',
    #                                                         'P2_SFA_0002', 'P1_FA_0008'])

    # res = compute_cv_for_samples_types(data.T, sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                                    'P2_SFA_0001', 'P2_SRM_0001',
    #                                                                    'P2_SFA_0002', 'P1_FA_0008'])
    # print(res)

    encodings = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/encodings.csv', index_col=0)

    # plot_full_dataset_umap(encodings, 'original samples', sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                                                 'P2_SFA_0001', 'P2_SRM_0001',
    #                                                                                 'P2_SFA_0002', 'P1_FA_0008'])

    res, _ = compute_number_of_clusters_with_hdbscan(encodings, print_info=True,
                                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                                            'P2_SFA_0002', 'P1_FA_0008'])
    print(res)







