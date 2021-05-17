

if __name__ == "__main__":

    # read config file
    config = {

        'data_path': '/Users/andreidm/ETH/projects/normalization/data/',
        'info_path': '/Users/andreidm/ETH/projects/normalization/data/',  # TODO: implement parsing batch info
        'out_path': '/Users/andreidm/ETH/projects/normalization/res/',

        'n_features': 170,  # TODO: infer from data
        'n_batches': 7,  # TODO: infer from batch info

        'latent_dim': -1,  # dimensions to reduce to (50 makes 99% of variance in PCA)
        'n_replicates': 3,

        'd_lr': 0.0014,  # TODO: implement intervals
        'g_lr': 0.0001,  # TODO: implement intervals
        'd_lambda': 8,  # TODO: implement intervals
        'g_lambda': 2.4,  # TODO: implement intervals

        'd_loss': 'CE',  # TODO: fix
        'g_loss': 'MSE',  # TODO: fix
        'use_g_regularization': True,  # TODO: fix

        'train_ratio': 0.9,  # TODO: make sure it is used where it's needed
        'batch_size': 64,
        'adversarial_epochs': 50,  # simultaneous competitive training

        'skip_epochs': 5,  # # TODO: test automatic skip, based on losses
        'callback_step': -1,  # TODO: test performance
        'keep_checkpoints': False  # whether to keep all checkpoints, or just the best epoch
    }


    # create grid

    # run grid

    # print best epochs


    pass