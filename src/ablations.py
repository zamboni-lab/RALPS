
import os, pandas, seaborn, numpy, traceback
import sys

from matplotlib import pyplot
from tqdm import tqdm
from pathlib import Path

from ralps import get_data, check_input, generate_parameters_grid, parse_config
from ralps import initialise_constant_parameters, get_grid_size
from models.adversarial import run_normalization
from evaluation import evaluate_models


def plot_lambdas(save_path=None):

    all_models = pandas.DataFrame()

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\lambdas_SRM_SPP\\v=0,d=0,g=0\\best_models.csv')
    best_models = best_models[best_models['rec_loss'] >= numpy.percentile(best_models['rec_loss'], 90)]
    best_models['lambdas'] = 'lv=0,\nld=0,\nlg=0'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\lambdas_SRM_SPP\\v=0,d=0\\best_models.csv')
    best_models = best_models[best_models['rec_loss'] >= numpy.percentile(best_models['rec_loss'], 90)]
    best_models['lambdas'] = 'lv=0,\nld=0,\nlg>0'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\lambdas_SRM_SPP\\v=0\\best_models.csv')
    best_models = best_models[best_models['rec_loss'] >= numpy.percentile(best_models['rec_loss'], 90)]
    best_models['lambdas'] = 'lv=0,\nld>0,\nlg>0'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\SRM+SPP\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['lambdas'] = 'lv>0,\nld>0,\nlg>0'
    all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='lambdas', y='reg_corr', data=all_models)
    pyplot.xlabel('Setting')
    pyplot.title('Mean cross-correlation of replicates')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'lambdas_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='lambdas', y='reg_grouping', data=all_models)
    pyplot.xlabel('Setting')
    pyplot.title('Mean grouping of replicates')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'lambdas_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='lambdas', y='batch_vc', data=all_models)
    pyplot.xlabel('Setting')
    pyplot.title('Mean batch VC')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'lambdas_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='lambdas', y='ivc_percent', data=all_models)
    pyplot.xlabel('Setting')
    pyplot.title('Mean percent of samples with increased VC')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'lambdas_vs_ivc_percent.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def plot_clustering(save_path=None):

    all_models = pandas.DataFrame()

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\birch\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'BIRCH'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\meanshift\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'MeanShift'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\optics\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'OPTICS'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\spectral\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'Spectral'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\upgma\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'UPGMA'
    all_models = pandas.concat([all_models, best_models])

    best_models = pandas.read_csv('D:\ETH\projects\\normalization\\res\SRM+SPP\\best_models.csv')
    best_models = best_models[best_models['best'] == True]
    best_models['algorithm'] = 'HDBSCAN'
    all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='algorithm', y='reg_corr', data=all_models)
    pyplot.title('Mean cross-correlation of replicates')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'clustering_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='algorithm', y='reg_grouping', data=all_models)
    pyplot.title('Mean grouping of replicates')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'clustering_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='algorithm', y='batch_vc', data=all_models)
    pyplot.title('Mean batch VC')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'clustering_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='algorithm', y='ivc_percent', data=all_models)
    pyplot.title('Mean percent of samples with increased VC')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'clustering_vs_ivc_percent.pdf')
        pyplot.close()

    timing = pandas.DataFrame({
        'algorithm': ['BIRCH', 'MeanShift', 'OPTICS', 'Spectral', 'UPGMA', 'HDBSCAN'],
        'time': [6., 10.7, 6.7, 6.2, 5.9, 5.4]})

    pyplot.figure()
    seaborn.barplot(x='algorithm', y='time', data=timing)
    pyplot.title('Mean RALPS run time (30 epochs)')
    pyplot.ylabel('Time, minutes')
    pyplot.tight_layout()
    if save_path is not None:
        pyplot.savefig(save_path + 'clustering_vs_time.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def plot_missing_values(save_path=None):

    common_path = 'D:\ETH\projects\\normalization\\res\\ablations_SRM_SPP\\na_fraction={}\\'

    all_models = pandas.DataFrame()
    for na_percent in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        best_models = pandas.read_csv(common_path.format(na_percent) + 'best_models.csv')
        best_models = best_models[best_models['best'] == True]
        best_models['NAs'] = na_percent
        all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='NAs', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of missing values')
    pyplot.title('Mean cross-correlation of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='NAs', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of missing values')
    pyplot.title('Mean grouping of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='NAs', y='batch_vc', data=all_models)
    pyplot.xlabel('Percent of missing values')
    pyplot.title('Mean batch VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='NAs', y='ivc_percent', data=all_models)
    pyplot.xlabel('Percent of missing values')
    pyplot.title('Mean percent of samples with increased VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_ivc_percent.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def plot_removed_metabolites(save_path=None):

    common_path = 'D:\ETH\projects\\normalization\\res\\ablations_SRM_SPP\\m_fraction={}\\'

    all_models = pandas.DataFrame()
    for percent in [1, 0.9, 0.7, 0.5, 0.3, 0.1]:
        best_models = pandas.read_csv(common_path.format(percent) + 'best_models.csv')
        best_models = best_models[best_models['best'] == True]
        best_models['m_percent'] = round(1-percent, 1)
        all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='m_percent', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of metabolites removed')
    pyplot.title('Mean cross-correlation of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='m_percent', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of metabolites removed')
    pyplot.title('Mean grouping of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='m_percent', y='batch_vc', data=all_models)
    pyplot.xlabel('Percent of metabolites removed')
    pyplot.title('Mean batch VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='m_percent', y='ivc_percent', data=all_models)
    pyplot.xlabel('Percent of metabolites removed')
    pyplot.title('Mean percent of samples with increased VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_ivc_percent.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def plot_removed_batches(save_path=None):

    common_path = 'D:\ETH\projects\\normalization\\res\\ablations_SRM_SPP\\n_batches={}\\'

    all_models = pandas.DataFrame()
    for n_batches in [7, 6, 5, 4, 3, 2]:
        best_models = pandas.read_csv(common_path.format(n_batches) + 'best_models.csv')
        best_models = best_models[best_models['best'] == True]
        best_models['n_batches'] = 7 - n_batches
        all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='n_batches', y='reg_corr', data=all_models)
    pyplot.xlabel('Number of batches removed')
    pyplot.title('Mean cross-correlation of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'n_batches_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='n_batches', y='reg_grouping', data=all_models)
    pyplot.xlabel('Number of batches removed')
    pyplot.title('Mean grouping of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'n_batches_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='n_batches', y='batch_vc', data=all_models)
    pyplot.xlabel('Number of batches removed')
    pyplot.title('Mean batch VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'n_batches_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='n_batches', y='ivc_percent', data=all_models)
    pyplot.xlabel('Number of batches removed')
    pyplot.title('Mean percent of samples with increased VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'n_batches_vs_ivc_percent.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def plot_variance_ratio(save_path=None):

    common_path = 'D:\ETH\projects\\normalization\\res\\ablations_SRM_SPP\\variance_ratio={}\\'

    all_models = pandas.DataFrame()
    ratios = [0.99, 0.95, 0.9, 0.85, 0.8, 0.7]
    for ratio in ratios:
        best_models = pandas.read_csv(common_path.format(ratio) + 'best_models.csv')
        best_models = best_models[best_models['best'] == True]
        best_models['variance_ratio'] = ratio
        all_models = pandas.concat([all_models, best_models])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.violinplot(x='variance_ratio', y='reg_corr', data=all_models, order=ratios)
    # pyplot.xlabel('Percent of explained variance in PCA')
    pyplot.title('Mean cross-correlation of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_corr.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='variance_ratio', y='reg_grouping', data=all_models, order=ratios)
    # pyplot.xlabel('Percent of explained variance in PCA')
    pyplot.title('Mean grouping of replicates')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_grouping.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='variance_ratio', y='batch_vc', data=all_models, order=ratios)
    # pyplot.xlabel('Percent of explained variance in PCA')
    pyplot.title('Mean batch VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_batch_vc.pdf')
        pyplot.close()

    pyplot.figure()
    seaborn.violinplot(x='variance_ratio', y='ivc_percent', data=all_models, order=ratios)
    # pyplot.xlabel('Percent of explained variance in PCA')
    pyplot.title('Mean percent of samples with increased VC')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_ivc_percent.pdf')
        pyplot.close()

    if save_path is None:
        pyplot.show()


def ablate_n_batches(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            new_config = config.copy()
            new_config['out_path'] += 'n_batches={}'.format(value)
            parameters = initialise_constant_parameters(new_config)
            data = get_data(new_config, parameters, n_batches=value)
            grid = generate_parameters_grid(get_grid_size(new_config), parameters, data)

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
                evaluate_models(new_config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
    else:
        print(warning)


def ablate_m_fraction(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            new_config = config.copy()
            new_config['out_path'] += 'm_fraction={}'.format(value)
            parameters = initialise_constant_parameters(new_config)
            data = get_data(new_config, parameters, m_fraction=value)
            grid = generate_parameters_grid(get_grid_size(new_config), parameters, data)

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
                evaluate_models(new_config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


def ablate_na_fraction(config, grid):
    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            new_config = config.copy()
            new_config['out_path'] += 'na_fraction={}'.format(value)
            parameters = initialise_constant_parameters(new_config)
            data = get_data(new_config, parameters, na_fraction=value)
            grid = generate_parameters_grid(get_grid_size(new_config), parameters, data)

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
                evaluate_models(new_config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


def ablate_variance_ratio(config, grid):

    is_correct, warning = check_input(config)
    if is_correct:

        for value in grid:
            new_config = config.copy()
            new_config['out_path'] += 'variance_ratio={}'.format(value)
            new_config['variance_ratio'] = str(value)
            parameters = initialise_constant_parameters(new_config)
            data = get_data(new_config, parameters)
            grid = generate_parameters_grid(get_grid_size(new_config), parameters, data)

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
                evaluate_models(new_config)
            except Exception as e:
                print('Ops! Error while evaluating models:\n', e)
        else:
            print(warning)


def run_ablations():
    """ This method is to be run from console.
        It gets a single config and runs ablations on it. """

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


if __name__ == "__main__":

    # save_path = 'D:\ETH\projects\\normalization\\res\\ablations_SRM_SPP\\plots\\'
    # plot_missing_values(save_path=save_path)
    # plot_removed_metabolites(save_path=save_path)
    # plot_removed_batches(save_path=save_path)
    # plot_variance_ratio(save_path=save_path)

    save_path = 'D:\ETH\projects\\normalization\\res\lambdas_SRM_SPP\\plots\\'
    plot_lambdas(save_path=save_path)
    # save_path = 'D:\ETH\projects\\normalization\\res\clustering_SRM_SPP\\plots\\'
    # plot_clustering(save_path=save_path)