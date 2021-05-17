
from src.models.adversarial import get_data, run_normalization

if __name__ == "__main__":

    # read config file
    config = {

        'data_path': '/Users/andreidm/ETH/projects/normalization/data/filtered_data.csv',
        'info_path': '/Users/andreidm/ETH/projects/normalization/data/batch_info.csv',  # TODO: update parsing batch info
        'out_path': '/Users/andreidm/ETH/projects/normalization/res/',

        'n_features': 170,  # TODO: infer from data
        'n_batches': 7,  # TODO: infer from batch info

        'latent_dim': -1,  # dimensions to reduce to (50 makes 99% of variance in PCA)
        'n_replicates': 3,

        'd_lr': 0.0014,  # TODO: implement intervals
        'g_lr': 0.0001,  # TODO: implement intervals
        'd_lambda': 8,  # TODO: implement intervals
        'g_lambda': 2.4,  # TODO: implement intervals

        'train_ratio': 0.9,  # TODO: make sure it is used where it's needed
        'batch_size': 64,
        'epochs': 3,  # simultaneous competitive training

        'skip_epochs': 5,  # # TODO: test automatic skip, based on losses
        'callback_step': 1,  # TODO: test performance
        'keep_checkpoints': True  # whether to keep all checkpoints, or just the best epoch
    }

    data = get_data(config['data_path'], config['info_path'])

    # create grid
    parameters = config.copy()
    parameters['id'] = 'noid'

    # run grid
    run_normalization(data, parameters)

    # print best epochs


    pass