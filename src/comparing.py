
import numpy, pandas

from src.models.adversarial import get_data
from src.batch_analysis import plot_batch_cross_correlations
from src.utils import combat


def plot_correlations_for_combat():
    path = '/Users/dmitrav/ETH/projects/normalization/data/'
    data, _ = get_data(path)

    combat_normalized = combat.combat(data.iloc[:, 1:].T, data['batch'])

    plot_batch_cross_correlations(combat_normalized.T, 'combat', '',
                                  sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                            'P2_SFA_0001', 'P2_SRM_0001',
                                                            'P2_SFA_0002', 'P1_FA_0008'])


if __name__ == "__main__":


    pass