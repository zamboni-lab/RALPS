
import pandas
from src.models.adversarial import get_data, run_normalization

if __name__ == "__main__":

    # read config file
    config = {

        'data_path': '/Users/andreidm/ETH/projects/normalization/data/filtered_data_v4.csv',
        'info_path': '/Users/andreidm/ETH/projects/normalization/data/batch_info_v41.csv',
        'out_path': '/Users/andreidm/ETH/projects/normalization/res/',

        'latent_dim': -1,  # dimensions to reduce to (50 makes 99% of variance in PCA)
        'n_replicates': 3,

        'd_lr': 0.0014,  # TODO: implement intervals
        'g_lr': 0.0001,  # TODO: implement intervals
        'd_lambda': 8,  # TODO: implement intervals
        'g_lambda': 2.4,  # TODO: implement intervals

        'train_ratio': 0.9,  # TODO: make sure it is used where it's needed
        'batch_size': 64,
        'epochs': 3,  # simultaneous competitive training

        'skip_epochs': 5,  # # TODO: implement and test automatic skip, based on losses
        'callback_step': -1,  # TODO: test performance
        'keep_checkpoints': True  # whether to keep all checkpoints, or just the best epoch
    }

    data = get_data(config['data_path'], config['info_path'])

    # parse parameters and create grid
    parameters = config.copy()
    parameters['n_features'] = data.shape[1]-1
    parameters['n_batches'] = data['batch'].unique().shape[0]

    reg_types = set()
    benchmarks = set()
    for i in range(data.index.shape[0]):
        if 'group_' in data.index[i]:
            reg_types.add('group_{}'.format(data.index[i].split('group')[1].split('_')[1]))
        if 'bench_' in data.index[i]:
            benchmarks.add('bench_{}'.format(data.index[i].split('bench')[1].split('_')[1]))

    parameters['reg_types'] = ','.join(list(reg_types))
    parameters['benchmarks'] = ','.join(list(benchmarks))

    parameters['id'] = 'noid'

    # run grid
    run_normalization(data, parameters)

    # print best epochs


    pass