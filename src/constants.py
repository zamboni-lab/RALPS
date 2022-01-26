
version = "v.0.6.22"

# CONFIG
default_parameters_values = {
    'latent_dim': -1,
    'variance_ratio': 0.99,
    'n_replicates': 3,
    'grid_size': 1,
    'epochs': 30,
    'skip_epochs': 5,
    'callback_step': -1,
    'train_ratio': 0.9,
    'keep_checkpoints': False,
    'plots_extension': 'png',
    'device': 'cpu',
    'min_relevant_intensity': 1000
}

# DATA
default_labels = ('0', '')  # for batch info

# MODEL SELECTION
grouping_threshold_percent = 30
correlation_threshold_percent = 70
n_best_models = 10