
import numpy, pandas

from src.models.adversarial import get_data
from src.batch_analysis import plot_batch_cross_correlations, compute_cv_for_samples_types
from src.batch_analysis import compute_number_of_clusters_with_hdbscan
from src.utils import combat


def plot_correlations_for_combat():
    path = '/Users/dmitrav/ETH/projects/normalization/data/'
    data, _ = get_data(path)

    combat_normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])

    plot_batch_cross_correlations(combat_normalized.T, 'combat', '',
                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                            'P2_SFA_0002', 'P1_FA_0008'])


def get_cvs_for_combat():

    path = '/Users/dmitrav/ETH/projects/normalization/data/'
    data, _ = get_data(path)

    combat_normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])

    res = compute_cv_for_samples_types(data.iloc[:, 1:], sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                                   'P2_SFA_0001', 'P2_SRM_0001',
                                                                                   'P2_SFA_0002', 'P1_FA_0008'])
    print(res)

    res = compute_cv_for_samples_types(combat_normalized.T, sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                                      'P2_SFA_0001', 'P2_SRM_0001',
                                                                                      'P2_SFA_0002', 'P1_FA_0008'])
    print(res)

    # TODO: plot bars for CVs


if __name__ == "__main__":

    path = '/Users/dmitrav/ETH/projects/normalization/data/'
    data, _ = get_data(path)

    combat_normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])

    pars = {'latent_dim': data.shape[1], 'n_batches': 7, 'n_replicates': 3}
    benchmarks = ['P1_FA_0001', 'P2_SF_0001', 'P2_SFA_0001', 'P2_SRM_0001', 'P2_SFA_0002', 'P1_FA_0008']

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