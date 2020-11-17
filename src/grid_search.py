
import pandas, numpy, uuid, os, random, sys, torch, time
from tqdm import tqdm
from multiprocessing import Process, Pool
from matplotlib import pyplot

from src.models import adversarial
from src.models.ae import Autoencoder
from src.batch_analysis import plot_batch_cross_correlations, plot_full_dataset_umap
from src.constants import samples_with_strong_batch_effects as benchmarks
from src.constants import user


def run_parallel(grid):
    """ Not working because of some weird Catalina error. """

    chunks = []
    for i in range(0, grid.shape[0], 3):
        chunk = []
        for j in range(i, i+3):

            if j >= grid.shape[0]:
                pass
            else:
                parameters = dict(grid.iloc[i,:])
                parameters['id'] = str(uuid.uuid4())[:8]
                chunk.append(parameters)
        chunks.append(chunk)

    for chunk in chunks:

        p1 = Process(target=adversarial.main, args=(chunk[0],))
        p2 = Process(target=adversarial.main, args=(chunk[1],))
        p3 = Process(target=adversarial.main, args=(chunk[2],))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()


def generate_random_parameter_set(g_loss, regularization, grid_size, grid_name, save_to):
    """ Generate a random parameter set to find out best parameter sets. """

    grid = range(grid_size)
    latent_dim = 50

    parameters = {

        'in_path': ['/Users/{}/ETH/projects/normalization/data/'.format(user) for x in grid],
        'out_path': ['/Users/{}/ETH/projects/normalization/res/grid_search/'.format(user) for x in grid],
        'id': [str(uuid.uuid4())[:8] for x in grid],

        'n_features': [170 for x in grid],  # n of metabolites in initial dataset
        'latent_dim': [latent_dim for x in grid],  # n dimensions to reduce to
        'n_batches': [7 for x in grid],
        'n_replicates': [3 for x in grid],

        'd_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # discriminator learning rate
        'g_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # generator learning rate
        'd_loss': ['CE' for x in grid],
        'g_loss': [g_loss for x in grid],
        'd_lambda': [round(random.uniform(0.1, 5), 1) for x in grid],  # discriminator regularization term coefficient
        'g_lambda':  [round(random.uniform(0.1, 5), 1) for x in grid],  # generator regularization term coefficient
        'use_g_regularization': [regularization for x in grid],  # whether to use generator regularization term
        'train_ratio': [0.9 for x in grid],  # for train-test split
        'batch_size': [64 for x in grid],
        'g_epochs': [0 for x in grid],  # pretraining of generator
        'd_epochs': [0 for x in grid],  # pretraining of discriminator
        'adversarial_epochs': [200 for x in grid],  # simultaneous competitive training

        'callback_step': [-1 for x in grid],  # save callbacks every n epochs
        'keep_checkpoints': [False for x in grid]  # whether to keep all checkpoints, or just the best epoch
    }

    grid = pandas.DataFrame(parameters)
    grid.to_csv(save_to + 'grid_{}_{}.csv'.format(grid_name, latent_dim))
    print('grid {} saved'.format(grid_name))


def generate_repetitive_grid(parameters_dict, grid_size, grid_name, save_to):
    """ Generate a grid with same parameters to find out how stable results are. """

    grid = range(grid_size)

    parameters = {}
    for key in parameters_dict.keys():
        if key == 'id' and parameters_dict[key] == '':
            parameters[key] = [str(uuid.uuid4())[:8] for x in grid]
        else:
            parameters[key] = [parameters_dict[key] for x in grid]

    grid = pandas.DataFrame(parameters)
    grid.to_csv(save_to + 'grid_{}.csv'.format(grid_name))
    print('grid {} saved'.format(grid_name))


def generate_and_save_repetitive_grids():
    """ Two best parameters sets and two approximations of them are used to generate repetitve grids. """

    save_to = '/Users/{}/ETH/projects/normalization/data/'.format(user)
    grid_size = 10

    grid_template = {

        'in_path': '/Users/{}/ETH/projects/normalization/data/'.format(user),
        'out_path': '/Users/{}/ETH/projects/normalization/res/grid_search/'.format(user),
        'id': '',

        'n_features': 170,  # n of metabolites in initial dataset
        'latent_dim': 100,  # n dimensions to reduce to
        'n_batches': 7,

        'd_lr': None,  # discriminator learning rate
        'g_lr': None,  # generator learning rate
        'd_loss': 'CE',
        'g_loss': None,
        'd_lambda': None,  # discriminator regularization term coefficient
        'g_lambda': None,  # generator regularization term coefficient
        'use_g_regularization': None,  # whether to use generator regularization term
        'train_ratio': 0.7,  # for train-test split
        'batch_size': 64,
        'g_epochs': 0,  # pretraining of generator
        'd_epochs': 0,  # pretraining of discriminator
        'adversarial_epochs': 300,  # simultaneous competitive training

        'callback_step': 1,  # save callbacks every n epochs
        'keep_checkpoints': True  # whether to keep all checkpoints, or just the best epoch
    }

    # set the best regularized parameter set
    best_l1_reg_grid = grid_template.copy()
    best_l1_reg_grid['d_lr'] = 0.0046
    best_l1_reg_grid['g_lr'] = 0.0042
    best_l1_reg_grid['g_loss'] = 'L1'
    best_l1_reg_grid['d_lambda'] = 3.3
    best_l1_reg_grid['g_lambda'] = 1.2
    best_l1_reg_grid['use_g_regularization'] = True

    generate_repetitive_grid(best_l1_reg_grid, grid_size, 'l1_reg_best', save_to)

    # set the approximation of the best regularized parameter set
    approx_best_l1_reg_grid = grid_template.copy()
    approx_best_l1_reg_grid['d_lr'] = 0.004
    approx_best_l1_reg_grid['g_lr'] = 0.004
    approx_best_l1_reg_grid['g_loss'] = 'L1'
    approx_best_l1_reg_grid['d_lambda'] = 3.
    approx_best_l1_reg_grid['g_lambda'] = 1.
    approx_best_l1_reg_grid['use_g_regularization'] = True

    generate_repetitive_grid(approx_best_l1_reg_grid, grid_size, 'l1_reg_approx_best', save_to)

    # set the best parameter set without regularization
    best_l1_grid = grid_template.copy()
    best_l1_grid['d_lr'] = 0.001
    best_l1_grid['g_lr'] = 0.0049
    best_l1_grid['g_loss'] = 'L1'
    best_l1_grid['d_lambda'] = .4
    best_l1_grid['g_lambda'] = 1.1
    best_l1_grid['use_g_regularization'] = False

    generate_repetitive_grid(best_l1_grid, grid_size, 'l1_best', save_to)

    # set the approximation of the best parameter set without regularization
    approx_best_l1_grid = grid_template.copy()
    approx_best_l1_grid['d_lr'] = 0.001
    approx_best_l1_grid['g_lr'] = 0.005
    approx_best_l1_grid['g_loss'] = 'L1'
    approx_best_l1_grid['d_lambda'] = .5
    approx_best_l1_grid['g_lambda'] = 1.
    approx_best_l1_grid['use_g_regularization'] = False

    generate_repetitive_grid(approx_best_l1_grid, grid_size, 'l1_approx_best', save_to)


def generate_random_grids():

    save_to = '/Users/{}/ETH/projects/normalization/data/'.format(user)
    generate_random_parameter_set('L1', True, 100, 'l1_reg', save_to)
    generate_random_parameter_set('L1', False, 100, 'l1', save_to)
    generate_random_parameter_set('SL1', True, 100, 'sl1_reg', save_to)
    # generate_random_parameter_set('SL1', False, 100, 'sl1', save_to)
    generate_random_parameter_set('MSE', True, 100, 'mse_reg', save_to)
    # generate_random_parameter_set('MSE', False, 100, 'mse', save_to)


def run_grid_from_console():
    """ To run from terminal with a single parameter: a grid file name. """
    name = sys.argv[1]

    path = '/Users/{}/ETH/projects/normalization/data/'.format(user)
    grid = pandas.read_csv(path + name, index_col=0)

    for i in tqdm(range(grid.shape[0])):
    # for i in tqdm(range(10, 30)):
        parameters = dict(grid.iloc[i, :])
        adversarial.main(parameters)
        time.sleep(600)


def collect_results_of_grid_search():
    """ Collect history files for all the grids for manual inspection. """

    grids_path = '/Users/{}/ETH/projects/normalization/data/grids/'.format(user)
    results_path = '/Users/{}/ETH/projects/normalization/res/grid_search/'.format(user)

    grids = ['grid_l1_reg_50', 'grid_l1_50', 'grid_mse_reg_50', 'grid_sl1_reg_50']
    results = {}

    for grid in grids:

        grid_pars = pandas.read_csv(grids_path + grid + '.csv', index_col=0)
        ids = grid_pars['id'].values

        best_epochs = pandas.DataFrame()
        for id in ids:

            id_results = pandas.read_csv(results_path + id + '/history_{}.csv'.format(id))
            id_results = id_results.loc[id_results['best'] == True, :]
            id_results['id'] = id

            best_epochs = pandas.concat([best_epochs, id_results])
            del id_results

        results[grid] = best_epochs

    for grid in results:
        print('GRID:', grid, '\n')
        top = adversarial.slice_by_grouping_and_correlation(results[grid], 30, 70)
        print(top.to_string(), '\n')

    return results


if __name__ == "__main__":

    # generate_random_grids()
    # run_grid_from_console()

    results = collect_results_of_grid_search()