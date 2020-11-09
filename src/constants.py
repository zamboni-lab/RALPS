
from torch import nn

version = "v.0.1.20"

allowed_ppm_error = 5
tic_normalization_scaling_factor = 10 ** 5
experiment_name_delimeter = '#'
number_of_replicates = 3

data_path = '/Users/andreidm/ETH/projects/normalization/data/'

# PARAMETERS

loss_mapper = {'CE': nn.CrossEntropyLoss(), 'L1': nn.L1Loss(), 'MSE': nn.MSELoss(), 'SL1': nn.SmoothL1Loss()}

# DATA

amino_acids = [
        ["Alanine", "C3H7NO2"],
        ["Arginine", "C6H14N4O2"],
        ["Asparagine", "C4H8N2O3"],
        ["Aspartate", "C4H7NO4"],
        ["Cysteine", "C3H7NO2S"],
        ['Glutamine', "C5H10N2O3"],
        ["Glutamate", "C5H9NO4"],  # Glutamic acid
        ["Glycine", "C2H5NO2"],
        ["Histidine", "C6H9N3O2"],
        ["Isoleucine", "C6H13NO2"],
        ["Leucine", "C6H13NO2"],
        ["Lysine", "C6H14N2O2"],
        ["Methionine", "C5H11NO2S"],
        ["Phenylalanine", "C9H11NO2"],
        ["Proline", "C5H9NO2"],
        ["Serine", "C3H7NO3"],
        ["Threonine", "C4H9NO3"],
        ["Tryptophan", "C11H12N2O2"],
        ["Tyrosine", "C9H11NO3"],
        ["Valine", "C5H11NO2"]
    ]

batches = ['0108', '0110', '0124', '0219', '0221', '0304', '0306']

# list of perturbations appearing in all 7 batches
shared_perturbations = ['P2_SRM_0001', 'P1_PP_0256', 'P2_SPP_0008', 'P2_SB_0008', 'P1_PP_2048', 'P1_Bio_0064', 'P1_PP_0004', 'P2_SB_0128', 'P2_FULL_0004', 'P2_SB_0032', 'P1_PP_0032', 'P2_SPP_0128', 'P1_FA_0004', 'P1_PP_0001', 'P2_FULL_0001', 'P2_SAA_0128', 'P1_Full_0008', 'P2_SFA_0016', 'P1_Full_0016', 'P2_SRM_0008', 'P1_Bio_1024', 'P2_SFA_0004', 'P1_SRM_0004', 'P1_AA_0256', 'P1_SRM_0001', 'P1_Bio_0016', 'P2_SRM_0002', 'P1_AA_2048', 'P1_PP_0512', 'P2_SPP_0002', 'P2_SFA_0128', 'P1_FA_0001', 'P1_SRM_0008', 'P1_SRM_0512', 'P1_PP_0008', 'P1_PP_1024', 'P2_SRM_0064', 'P1_Bio_0032', 'P2_SFA_0002', 'P1_Full_0064', 'P1_Full_0004', 'P2_SRM_0032', 'P2_SF_0001', 'P1_PP_0002', 'P1_FA_2048', 'P1_Full_0001', 'P1_AA_1024', 'P2_SRM_2048', 'P2_FULL_0002', 'P2_SAA_0032', 'P1_AA_0512', 'P2_SF_0004', 'P1_Full_0002', 'P1_FA_0016', 'P1_FA_0128', 'P2_SB_0004', 'P1_PP_0016', 'P1_Full_0128', 'P1_FA_0064', 'P1_SRM_0256', 'P2_SF_0128', 'P2_FULL_0256', 'P2_SFA_0032', 'P2_SPP_0001', 'P1_Bio_0256', 'P1_SRM_0002', 'P2_FULL_2048', 'P2_SF_0064', 'P1_PP_0064', 'P1_AA_0064', 'P2_SAA_0002', 'P2_SF_0016', 'P1_Full_0032', 'P1_Bio_0512', 'P1_Full_0256', 'P2_SF_0002', 'P2_SAA_0064', 'P1_SRM_0128', 'P2_SAA_0016', 'P2_SFA_0064', 'P1_SRM_2048', 'P2_SRM_0004', 'P1_AA_0002', 'P1_AA_0008', 'P2_FULL_0008', 'P1_AA_0004', 'P1_SRM_0016', 'P1_Bio_0002', 'P2_FULL_1024', 'P2_SPP_0032', 'P2_SRM_0512', 'P1_AA_0128', 'P2_SAA_0004', 'P1_Bio_0128', 'P1_Bio_0004', 'P1_Full_1024', 'P2_SRM_0128', 'P1_FA_0256', 'P2_SB_0016', 'P1_SRM_1024', 'P2_FULL_0032', 'P2_SRM_1024', 'P2_SAA_0008', 'P2_FULL_0064', 'P1_SRM_0032', 'P1_FA_0032', 'P1_AA_0001', 'P2_FULL_0016', 'P1_AA_0016', 'P2_SFA_0001', 'P2_SRM_0256', 'P2_SPP_0016', 'P1_SRM_0064', 'P2_SRM_0016', 'P1_FA_0512', 'P1_Full_2048', 'P2_SF_0008', 'P2_FULL_0128', 'P2_SF_0032', 'P2_SB_0002', 'P1_FA_0002', 'P1_Full_0512', 'P1_Bio_0008', 'P2_FULL_0512', 'P1_PP_0128', 'P2_SAA_0001', 'P2_SB_0064', 'P1_FA_1024', 'P1_FA_0008', 'P2_SPP_0004', 'P2_SB_0001', 'P1_Bio_0001', 'P1_AA_0032', 'P2_SPP_0064', 'P1_Bio_2048', 'P2_SFA_0008']

# list of samples to be used as controls for NormAE
controls = ['P1_PP_0001', 'P1_PP_0002', 'P1_PP_0004', 'P1_PP_0008']  # example

# samples that show noticeable batch effects, can be taken for benchmarking
samples_with_strong_batch_effects = ['P1_FA_0001', 'P2_SF_0001', 'P2_SFA_0001', 'P2_SRM_0001', 'P2_SFA_0002', 'P1_FA_0008']
