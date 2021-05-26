
from torch import nn

# META
version = "v.0.3.48"

default_parameters_values = {
    'latent_dim': -1,
    'n_replicates': 3,
    'grid_size': 1,
    'epochs': 50,
    'skip_epochs': 5,
    'callback_step': -1,
    'train_ratio': 0.9,
    'keep_checkpoints': False
}

user = 'andreidm'

data_path = '/Users/{}/ETH/projects/normalization/data/'.format(user)

# SCENARIO #1: NO REFERENCE SAMPLES
path_to_my_best_method_1 = '/Users/{}/ETH/projects/normalization/res/no_reference_samples/best_model/2d48bfb2/normalized_2d48bfb2.csv'.format(user)
path_to_other_methods_1 = '/Users/{}/ETH/projects/normalization/res/no_reference_samples/other_methods/'.format(user)

# SCENARIO #2: WITH 8 SRMs AS REFERENCE SAMPLES
path_to_my_best_method_2 = '/Users/{}/ETH/projects/normalization/res/fake_reference_samples/grid_656cfcf3/11bf6f68/normalized_11bf6f68.csv'.format(user)
path_to_other_methods_2 = '/Users/{}/ETH/projects/normalization/res/fake_reference_samples/other_methods/'.format(user)

# DATA
min_relevant_intensity = 1000
latent_dim_explained_variance_ratio = 0.99

batches = ['0108', '0110', '0124', '0219', '0221', '0304', '0306']

# list of perturbations appearing in all 7 batches
shared_perturbations = ['P2_SRM_0001', 'P1_PP_0256', 'P2_SPP_0008', 'P2_SB_0008', 'P1_PP_2048', 'P1_Bio_0064', 'P1_PP_0004', 'P2_SB_0128', 'P2_FULL_0004', 'P2_SB_0032', 'P1_PP_0032', 'P2_SPP_0128', 'P1_FA_0004', 'P1_PP_0001', 'P2_FULL_0001', 'P2_SAA_0128', 'P1_Full_0008', 'P2_SFA_0016', 'P1_Full_0016', 'P2_SRM_0008', 'P1_Bio_1024', 'P2_SFA_0004', 'P1_SRM_0004', 'P1_AA_0256', 'P1_SRM_0001', 'P1_Bio_0016', 'P2_SRM_0002', 'P1_AA_2048', 'P1_PP_0512', 'P2_SPP_0002', 'P2_SFA_0128', 'P1_FA_0001', 'P1_SRM_0008', 'P1_SRM_0512', 'P1_PP_0008', 'P1_PP_1024', 'P2_SRM_0064', 'P1_Bio_0032', 'P2_SFA_0002', 'P1_Full_0064', 'P1_Full_0004', 'P2_SRM_0032', 'P2_SF_0001', 'P1_PP_0002', 'P1_FA_2048', 'P1_Full_0001', 'P1_AA_1024', 'P2_SRM_2048', 'P2_FULL_0002', 'P2_SAA_0032', 'P1_AA_0512', 'P2_SF_0004', 'P1_Full_0002', 'P1_FA_0016', 'P1_FA_0128', 'P2_SB_0004', 'P1_PP_0016', 'P1_Full_0128', 'P1_FA_0064', 'P1_SRM_0256', 'P2_SF_0128', 'P2_FULL_0256', 'P2_SFA_0032', 'P2_SPP_0001', 'P1_Bio_0256', 'P1_SRM_0002', 'P2_FULL_2048', 'P2_SF_0064', 'P1_PP_0064', 'P1_AA_0064', 'P2_SAA_0002', 'P2_SF_0016', 'P1_Full_0032', 'P1_Bio_0512', 'P1_Full_0256', 'P2_SF_0002', 'P2_SAA_0064', 'P1_SRM_0128', 'P2_SAA_0016', 'P2_SFA_0064', 'P1_SRM_2048', 'P2_SRM_0004', 'P1_AA_0002', 'P1_AA_0008', 'P2_FULL_0008', 'P1_AA_0004', 'P1_SRM_0016', 'P1_Bio_0002', 'P2_FULL_1024', 'P2_SPP_0032', 'P2_SRM_0512', 'P1_AA_0128', 'P2_SAA_0004', 'P1_Bio_0128', 'P1_Bio_0004', 'P1_Full_1024', 'P2_SRM_0128', 'P1_FA_0256', 'P2_SB_0016', 'P1_SRM_1024', 'P2_FULL_0032', 'P2_SRM_1024', 'P2_SAA_0008', 'P2_FULL_0064', 'P1_SRM_0032', 'P1_FA_0032', 'P1_AA_0001', 'P2_FULL_0016', 'P1_AA_0016', 'P2_SFA_0001', 'P2_SRM_0256', 'P2_SPP_0016', 'P1_SRM_0064', 'P2_SRM_0016', 'P1_FA_0512', 'P1_Full_2048', 'P2_SF_0008', 'P2_FULL_0128', 'P2_SF_0032', 'P2_SB_0002', 'P1_FA_0002', 'P1_Full_0512', 'P1_Bio_0008', 'P2_FULL_0512', 'P1_PP_0128', 'P2_SAA_0001', 'P2_SB_0064', 'P1_FA_1024', 'P1_FA_0008', 'P2_SPP_0004', 'P2_SB_0001', 'P1_Bio_0001', 'P1_AA_0032', 'P2_SPP_0064', 'P1_Bio_2048', 'P2_SFA_0008']

# sample types that are used to evaluate normalization
benchmark_sample_types = ['P1_FA_0001', 'P2_SF_0001', 'P2_SFA_0001', 'P2_SRM_0001', 'P2_SFA_0002', 'P1_FA_0008']

# sample types that are used for regularization

# # SCENARIO #1: copy all perturbations for completely untargeted case
# regularization_sample_types = shared_perturbations[:]

# # SCENARIO #2:  use 8 SRM samples in each batch as references (internal standards)
# regularization_sample_types = [x for x in shared_perturbations if 'SRM_000' in x]

# # SCENARIO #2 with the other reference samples:
regularization_sample_types = ['P2_SRM_0001', 'P2_SRM_0002', 'P2_SRM_0004']
# regularization_sample_types = ['P2_SRM_0001', 'P2_SRM_0002', 'P2_SRM_0004', 'P2_SRM_0008']

# regularization_sample_types = ['P2_SRM_0001', 'P2_SPP_0001']
# regularization_sample_types = ['P2_SRM_0001', 'P1_SRM_0001', 'P2_SRM_0002']
# regularization_sample_types = ['P2_SRM_0001', 'P1_SRM_0001', 'P2_SRM_0002', 'P1_SRM_0002']

# regularization_sample_types = ['P2_SRM_0001', 'P2_SRM_0002', 'P2_SPP_0001', 'P2_SPP_0002']  # with 504c09ce
# regularization_sample_types = ['P2_SRM_0001', 'P2_SF_0001']  # with 04b7b4ac
# regularization_sample_types = ['P2_SRM_0001', 'P2_SFA_0001']  # with 6a12d914