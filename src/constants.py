
version = "v.0.6.47"

# CONFIG
required_config_fields = ['data_path', 'info_path', 'out_path', 'latent_dim', 'variance_ratio', 'n_replicates',
                          'grid_size', 'd_lr', 'g_lr', 'd_lambda', 'g_lambda', 'v_lambda', 'train_ratio', 'batch_size',
                          'epochs', 'skip_epochs', 'keep_checkpoints', 'device', 'plots_extension',
                          'min_relevant_intensity', 'allowed_vc_increase']

default_parameters_values = {
    'latent_dim': -1,
    'variance_ratio': 0.99,
    'n_replicates': 3,
    'grid_size': 1,
    'epochs': 30,
    'skip_epochs': 3,
    'train_ratio': 0.9,
    'keep_checkpoints': False,
    'plots_extension': 'png',
    'device': 'cpu',
    'min_relevant_intensity': 1000,
    'allowed_vc_increase': 0.05
}

default_parameters_ranges = {
    'd_lr': [0.00005, 0.005],
    'g_lr': [0.00005, 0.005],
    'd_lambda': [0., 10.],
    'g_lambda': [0., 10.],
    'v_lambda': [0., 10.],
    'batch_size': [32, 64, 128],
    'variance_ratio': [0.9, 0.95, 0.99]
}

# DATA
default_labels = ('0', '')  # for batch info

# REGULARIZATION
clustering_algorithm = 'hdbscan'  # 'hdbscan', 'upgma', 'mean_shift', 'optics', 'birch', 'spectral'
clustering_metric = 'braycurtis'

# MODEL SELECTION
grouping_threshold_percent = 30
correlation_threshold_percent = 70
n_best_models = 10