
version = "v.0.4.4"

# CONFIG
default_parameters_values = {
    'latent_dim': -1,
    'n_replicates': 3,
    'grid_size': 1,
    'epochs': 50,
    'skip_epochs': 5,
    'callback_step': -1,
    'train_ratio': 0.9,
    'keep_checkpoints': False,
    'min_relevant_intensity': 1000
}

# DATA
latent_dim_explained_variance_ratio = 0.99

# MODEL SELECTION
grouping_threshold_percent = 30
correlation_threshold_percent = 70