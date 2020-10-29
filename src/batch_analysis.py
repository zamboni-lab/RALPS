
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

    samples_by_types = get_samples_by_types_dict(data.columns.values, sample_types_of_interest)

    if sample_types_of_interest is None:
        for i, type in enumerate(samples_by_types):
            df = data.loc[:, numpy.array(samples_by_types[type])]
            df.columns = [x[-6:] for x in df.columns]
            df = df.corr()

            seaborn.heatmap(df)
            pyplot.title('Cross correlations: {}: {}'.format(type, method_name))
            pyplot.tight_layout()
            pyplot.show()

    else:
        pyplot.figure(figsize=(12,8))

        for i, type in enumerate(samples_by_types):
            df = data.loc[:, numpy.array(samples_by_types[type])]
            df.columns = [x[-6:] for x in df.columns]
            df = df.corr()

            ax = pyplot.subplot(2, 3, i+1)
            seaborn.heatmap(df)
            ax.set_title(type)

        pyplot.suptitle('Cross correlations: {}'.format(method_name))
        pyplot.tight_layout()
        # pyplot.show()
        pyplot.savefig(save_to + 'correlations_{}.pdf'.format(method_name.replace(' ', '_')))


def compute_cv_for_samples_types(data, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(data.columns.values, sample_types_of_interest)

    cv_dict = {}
    for i, type in enumerate(samples_by_types):
        values = data.loc[:, numpy.array(samples_by_types[type])].values
        values = values.flatten()
        cv_dict[type] = numpy.std(values) / numpy.mean(values)

    return cv_dict


def plot_batch_effects_with_umap(encodings, method_name, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    if sample_types_of_interest is None:
        sample_types_of_interest = ['P1_FA_0001', 'P2_SF_0001', 'P2_SFA_0001', 'P2_SRM_0001', 'P2_SFA_0002', 'P1_FA_0008']

    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    random_seed = 831
    metric = 'braycurtis'
    n = 20

    reducer = umap.UMAP(n_neighbors=n, metric=metric, min_dist=0.1, random_state=random_seed)
    embedding = reducer.fit_transform(values)

    seaborn.set()
    pyplot.figure(figsize=(12, 8))
    for i, type in enumerate(samples_by_types):
        indices_of_type = [list(encodings.index.values).index(x) for x in samples_by_types[type]]
        type_embedding = embedding[numpy.array(indices_of_type), :]
        type_batches = batches[numpy.array(indices_of_type)]

        ax = pyplot.subplot(2, 3, i + 1)
        seaborn.scatterplot(x=type_embedding[:, 0], y=type_embedding[:, 1], hue=type_batches, s=50, alpha=.8,
                            palette=seaborn.color_palette('colorblind', n_colors=len(set(type_batches))))
        ax.set_title(type)
        ax.get_legend().remove()

    pyplot.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    pyplot.suptitle('{}'.format(method_name, n, metric))
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(save_to + 'umap_batch_effects_{}.pdf'.format(method_name.replace(' ', '_')))


def compute_number_of_clusters_with_hdbscan(encodings, print_info=True, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    samples_by_types = get_samples_by_types_dict(encodings.index.values, sample_types_of_interest)

    batches = encodings['batch'].values - 1
    values = encodings.iloc[:, 1:].values

    random_seed = 831
    metric = 'braycurtis'
    n = 20

    reducer = umap.UMAP(n_components=30, n_neighbors=n, metric=metric, min_dist=0.1, random_state=random_seed)
    embeddings = reducer.fit_transform(values)

    n_clusters_dict = {}
    for i, type in enumerate(samples_by_types):
        indices_of_type = [list(encodings.index.values).index(x) for x in samples_by_types[type]]
        type_embeddings = embeddings[numpy.array(indices_of_type), :]
        type_batches = batches[numpy.array(indices_of_type)]

        clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=3, allow_single_cluster=True)
        clusterer.fit(type_embeddings)

        if print_info:
            print('CLUSTERING INFO:', type)
            print('n clusters: ', clusterer.labels_.max() + 1)
            print('labels:', list(clusterer.labels_))
            print('probs:', list(clusterer.probabilities_))
            print('outlier probs:', list(clusterer.outlier_scores_), '\n')

        n_clusters_dict[type] = clusterer.labels_.max() + 1

    return n_clusters_dict


if __name__ == '__main__':

    # data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data.csv')
    # data = data.iloc[:, 3:]

    # plot_batch_cross_correlations(data, 'original samples',
    #                               sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                         'P2_SFA_0001', 'P2_SRM_0001',
    #                                                         'P2_SFA_0002', 'P1_FA_0008'])

    # res = compute_cv_for_samples_types(data, sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                                    'P2_SFA_0001', 'P2_SRM_0001',
    #                                                                    'P2_SFA_0002', 'P1_FA_0008'])
    # print(res)

    encodings = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/encodings.csv', index_col=0)

    # plot_batch_effects_with_umap(encodings, 'original samples',
    #                              sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                        'P2_SFA_0001', 'P2_SRM_0001',
    #                                                        'P2_SFA_0002', 'P1_FA_0008'])

    res = compute_number_of_clusters_with_hdbscan(encodings, print_info=False,
                                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                                            'P2_SFA_0002', 'P1_FA_0008'])

    print(res)







