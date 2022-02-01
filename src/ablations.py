
import os, pandas, seaborn, numpy, traceback
import sys

from matplotlib import pyplot
from tqdm import tqdm
from pathlib import Path

from ralps import get_data, check_input, generate_parameters_grid, parse_config
from models.adversarial import run_normalization
from evaluation import evaluate_models


def choose_best_for_50_percent_NAs(best_models, g_percent, c_percent):

    try:
        # negative loss slice (desired by model construction)
        df = best_models[best_models['g_loss'] < 0]
        # grouping slice
        df = df[df['reg_grouping'] <= numpy.percentile(df['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        df['best'] = True
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def choose_best_for_2_batches(best_models, g_percent, c_percent):

    try:
        # slice by g_loss
        df = best_models[best_models['g_loss'] <= numpy.percentile(best_models['g_loss'].values, g_percent)]
        # grouping slice
        df = df[df['reg_grouping'] <= numpy.percentile(df['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        # negative loss slice (desired by model construction)

        df['best'] = True
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def plot_missing_values(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/7_batches_1.0_metabolites_{}_NAs/best_models.csv'

    all_models = pandas.DataFrame()
    for na_percent in [0.05, 0.1, 0.15, 0.2, 0.3]:
        best_models = pandas.read_csv(common_path.format(na_percent))
        best_models = best_models[best_models['best'] == True]
        best_models['NAs'] = na_percent
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_grouping.pdf')

    if save_path is None:
        pyplot.show()


def plot_removed_metabolites(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/7_batches_{}_metabolites/best_models.csv'

    all_models = pandas.DataFrame()
    for percent in [0.2, 0.4, 0.6, 0.8, 1.0]:
        best_models = pandas.read_csv(common_path.format(percent))
        best_models = best_models[best_models['best'] == True]
        best_models['percent'] = percent
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='percent', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='percent', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='percent', y='b_corr', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_b_corr.pdf')

    if save_path is None:
        pyplot.show()


def plot_removed_batches(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/{}_batches_1.0_metabolites/best_models.csv'

    all_models = pandas.DataFrame()
    for n in [2, 4, 7]:
        best_models = pandas.read_csv(common_path.format(n))
        if n == 2:
            best_models = choose_best_for_2_batches(best_models, 30, 70)
        else:
            best_models = best_models[best_models['best'] == True]
        best_models['n'] = n
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='n', y='reg_corr', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='reg_vc', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='b_corr', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_b_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='b_grouping', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_b_grouping.pdf')

    if save_path is None:
        pyplot.show()


def plot_variance_ratio(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/variance_ratio_P2_SRM_0001_0002_0004/{}/best_models.csv'

    all_models = pandas.DataFrame()
    for ratio in [0.7, 0.8, 0.9, 0.95, 0.99]:
        best_models = pandas.read_csv(common_path.format(ratio))
        best_models = best_models[best_models['best'] == True]
        best_models['variance_ratio'] = ratio
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_grouping.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='b_corr', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_b_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='b_grouping', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_b_grouping.pdf')

    if save_path is None:
        pyplot.show()


def ablate_n_batches(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            config['out_path'] += 'n_batches={}'.format(value)
            data = get_data(config, n_batches=value)
            grid = generate_parameters_grid(config, data)

            for parameters in tqdm(grid):
                try:
                    run_normalization(data, parameters)
                except Exception as e:
                    print("failed with", e)
                    log_path = Path(parameters['out_path']) / parameters['id'] / 'traceback.txt'
                    with open(log_path, 'w') as f:
                        f.write(traceback.format_exc())
                    print("full traceback saved to", log_path, '\n')

            print('Grid search completed.\n')
            try:
                evaluate_models(config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
    else:
        print(warning)


def ablate_m_fraction(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            config['out_path'] += 'm_fraction={}'.format(value)
            data = get_data(config, m_fraction=value)
            grid = generate_parameters_grid(config, data)

            for parameters in tqdm(grid):
                try:
                    run_normalization(data, parameters)
                except Exception as e:
                    print("failed with", e)
                    log_path = Path(parameters['out_path']) / parameters['id'] / 'traceback.txt'
                    with open(log_path, 'w') as f:
                        f.write(traceback.format_exc())
                    print("full traceback saved to", log_path, '\n')

            print('Grid search completed.\n')
            try:
                evaluate_models(config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


def ablate_na_fraction(config, grid):
    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            config['out_path'] += 'na_fraction={}'.format(value)
            data = get_data(config, na_fraction=value)
            grid = generate_parameters_grid(config, data)

            for parameters in tqdm(grid):
                try:
                    run_normalization(data, parameters)
                except Exception as e:
                    print("failed with", e)
                    log_path = Path(parameters['out_path']) / parameters['id'] / 'traceback.txt'
                    with open(log_path, 'w') as f:
                        f.write(traceback.format_exc())
                    print("full traceback saved to", log_path, '\n')

            print('Grid search completed.\n')
            try:
                evaluate_models(config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


def ablate_variance_ratio(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            config['out_path'] += 'variance_ratio={}'.format(value)
            config['variance_ratio'] = str(value)
            data = get_data(config)
            grid = generate_parameters_grid(config, data)

            for parameters in tqdm(grid):
                try:
                    run_normalization(data, parameters)
                except Exception as e:
                    print("failed with", e)
                    log_path = Path(parameters['out_path']) / parameters['id'] / 'traceback.txt'
                    with open(log_path, 'w') as f:
                        f.write(traceback.format_exc())
                    print("full traceback saved to", log_path, '\n')

            print('Grid search completed.\n')
            try:
                evaluate_models(config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


if __name__ == "__main__":

    config = parse_config()

    if sys.argv[2] == 'n_batches':
        ablate_n_batches(config, [7, 6, 5, 4, 3, 2])
    elif sys.argv[2] == 'm_fraction':
        ablate_m_fraction(config, [1, 0.9, 0.7, 0.5, 0.3, 0.1])
    elif sys.argv[2] == 'na_fraction':
        ablate_na_fraction(config, [0, 0.05, 0.1, 0.15, 0.2, 0.3])
    elif sys.argv[2] == 'variance_ratio':
        ablate_variance_ratio(config, [0.7, 0.8, 0.85, 0.9, 0.95, 0.99])
    else:
        raise ValueError('Ablation not recognized.')
