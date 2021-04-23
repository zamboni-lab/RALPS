
import pandas, numpy, uuid, os, random, sys, torch, time
from tqdm import tqdm
from multiprocessing import Process, Pool
from matplotlib import pyplot

from src.models import adversarial
from src.models.ae import Autoencoder
from src.batch_analysis import plot_batch_cross_correlations, plot_full_dataset_umap
from src.constants import benchmark_sample_types as benchmarks
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
    latent_dim = 50  # 50 principal components explain >95% of variance for Michelle's dataset

    parameters = {

        'in_path': ['/Users/{}/ETH/projects/normalization/data/'.format(user) for x in grid],
        'out_path': ['/Users/{}/ETH/projects/normalization/res/{}/grid_search/'.format(user, grid_name) for x in grid],
        'id': [str(uuid.uuid4())[:8] for x in grid],

        'n_features': [170 for x in grid],  # n of metabolites in initial dataset
        'latent_dim': [latent_dim for x in grid],  # n dimensions to reduce to
        'n_batches': [7 for x in grid],
        'n_replicates': [3 for x in grid],

        'd_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # discriminator learning rate
        'g_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # generator learning rate
        'd_loss': ['CE' for x in grid],
        'g_loss': [g_loss for x in grid],
        'd_lambda': [round(random.uniform(0., 10.), 1) for x in grid],  # discriminator regularization term coefficient
        'g_lambda':  [round(random.uniform(0., 10.), 1) for x in grid],  # generator regularization term coefficient
        'use_g_regularization': [regularization for x in grid],  # whether to use generator regularization term
        'train_ratio': [0.9 for x in grid],  # for train-test split
        'batch_size': [64 for x in grid],
        'g_epochs': [0 for x in grid],  # pretraining of generator (not yet implemented)
        'd_epochs': [0 for x in grid],  # pretraining of discriminator  (not yet implemented)
        'adversarial_epochs': [100 for x in grid],  # simultaneous competitive training

        'skip_epochs': [5 for x in grid],
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
    grid_size = 50

    grid_template = {

        'in_path': '/Users/{}/ETH/projects/normalization/data/'.format(user),
        'out_path': '/Users/{}/ETH/projects/normalization/res/P2_SRM_0001+P2_SFA_0001/grid_search/'.format(user),
        'id': '',

        'n_features': 170,  # n of metabolites in initial dataset
        'latent_dim': 50,  # n dimensions to reduce to (50 makes 99% of variance in PCA)
        'n_batches': 7,
        'n_replicates': 3,

        'd_lr': None,  # discriminator learning rate
        'g_lr': None,  # generator learning rate
        'd_loss': 'CE',
        'g_loss': 'MSE',
        'd_lambda': None,  # discriminator regularization term coefficient
        'g_lambda': None,  # generator regularization term coefficient
        'use_g_regularization': True,  # whether to use generator regularization term
        'train_ratio': 0.9,  # for train-test split
        'batch_size': 64,
        'g_epochs': 0,  # pretraining of generator
        'd_epochs': 0,  # pretraining of discriminator
        'adversarial_epochs': 50,  # simultaneous competitive training

        'skip_epochs': 5,
        'callback_step': -1,  # save callbacks every n epochs
        'keep_checkpoints': False  # whether to keep all checkpoints, or just the best epoch
    }

    # set the best regularized parameter set
    grid_6a12d914 = grid_template.copy()
    grid_6a12d914['d_lr'] = 0.0032
    grid_6a12d914['g_lr'] = 0.0025
    grid_6a12d914['d_lambda'] = 9.6
    grid_6a12d914['g_lambda'] = 7.0
    grid_6a12d914['out_path'] = grid_6a12d914['out_path'].replace('grid_search', 'grid_6a12d914')

    generate_repetitive_grid(grid_6a12d914, grid_size, 'SRM+SFA_6a12d914', save_to)

    # set the approximation of the best regularized parameter set
    grid_04b7b4ac = grid_template.copy()
    grid_04b7b4ac['d_lr'] = 0.0001
    grid_04b7b4ac['g_lr'] = 0.0008
    grid_04b7b4ac['d_lambda'] = 6.3
    grid_04b7b4ac['g_lambda'] = 1.1
    grid_04b7b4ac['out_path'] = grid_04b7b4ac['out_path'].replace('grid_search', 'grid_04b7b4ac')

    generate_repetitive_grid(grid_04b7b4ac, grid_size, 'SRM+SF_04b7b4ac', save_to)

    # set the best parameter set without regularization
    grid_ba596703 = grid_template.copy()
    grid_ba596703['d_lr'] = 0.0014
    grid_ba596703['g_lr'] = 0.0001
    grid_ba596703['d_lambda'] = 8.0
    grid_ba596703['g_lambda'] = 2.4
    grid_ba596703['out_path'] = grid_ba596703['out_path'].replace('grid_search', 'grid_ba596703')

    generate_repetitive_grid(grid_ba596703, grid_size, 'SRM+SPP_ba596703', save_to)


def generate_random_grids():

    save_to = '/Users/{}/ETH/projects/normalization/data/'.format(user)

    # # testing various parameters...
    # generate_random_parameter_set('L1', True, 100, 'l1_reg', save_to)
    # generate_random_parameter_set('L1', False, 100, 'l1', save_to)
    # generate_random_parameter_set('SL1', True, 100, 'sl1_reg', save_to)
    # generate_random_parameter_set('SL1', False, 100, 'sl1', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'new_mse_reg', save_to)
    # generate_random_parameter_set('MSE', False, 100, 'mse', save_to)

    # # for training with fake reference samples
    # generate_random_parameter_set('MSE', True, 100, 'fake_refs_mse_reg', save_to)

    # for training with fewer reference samples
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_SAA_0001', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_SB_0001', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_SFA_0001', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_SF_0001', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_Full_0001', save_to)
    # generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P2_SPP_0001', save_to)

    # for training with fewer reference samples
    generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P1_SRM_0001', save_to)
    generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P1_SRM_0001+P2_SRM_0002', save_to)
    generate_random_parameter_set('MSE', True, 100, 'P2_SRM_0001+P1_SRM_0001+P2_SRM_0002+P1_SRM_0002', save_to)


def run_grid_from_console():
    """ To run from terminal with a single parameter: a grid file name. """
    name = sys.argv[1]

    path = '/Users/{}/ETH/projects/normalization/data/'.format(user)
    grid = pandas.read_csv(path + name, index_col=0)

    for i in tqdm(range(grid.shape[0])):
    # for i in tqdm(range(67, 100)):
        parameters = dict(grid.iloc[i, :])
        adversarial.main(parameters)


def collect_results_of_grid_search(results_path, grid_name):
    """ Collect history files for all the grids for manual inspection. """

    grids_path = '/Users/{}/ETH/projects/normalization/data/grids/'.format(user)
    results = {}

    grid_pars = pandas.read_csv(grids_path + grid_name + '.csv', index_col=0)
    ids = grid_pars['id'].values

    best_epochs = pandas.DataFrame()
    for id in ids:

        id_results = pandas.read_csv(results_path + id + '/history_{}.csv'.format(id))
        id_results = id_results.loc[id_results['best'] == True, :]
        id_results['id'] = id

        best_epochs = pandas.concat([best_epochs, id_results])
        del id_results

    print('GRID:', grid_name, '\n')
    top = adversarial.slice_by_grouping_and_correlation(best_epochs, 30, 70)
    print(top.to_string(), '\n')

    return results


def collect_results_of_repetitive_runs(path):

    best_epochs = pandas.DataFrame()

    for id in os.listdir(path):
        if not id.startswith('.'):

            id_results = pandas.read_csv(path + id + '/history_{}.csv'.format(id))
            id_results = id_results.loc[id_results['best'] == True, :]
            id_results['id'] = id

            best_epochs = pandas.concat([best_epochs, id_results])
            del id_results

    print('Results for {}:'.format(path.split('/')[-2]))
    print(best_epochs.to_string(),'\n')

    return best_epochs


if __name__ == "__main__":

    # generate_random_grids()
    # generate_and_save_repetitive_grids()

    run_grid_from_console()
    # results = collect_results_of_grid_search('/Users/{}/ETH/projects/normalization/res/P2_SRM_0001+P2_SPP_0001/grid_search/'.format(user),
    #                                          'grid_P2_SRM_0001+P2_SPP_0001_50')

    # results = collect_results_of_repetitive_runs('/Users/{}/ETH/projects/normalization/res/fake_reference_samples/grid_656cfcf3/'.format(user))
    # results = collect_results_of_repetitive_runs('/Users/{}/ETH/projects/normalization/res/fake_reference_samples/grid_1657c7f8/'.format(user))
    # results = collect_results_of_repetitive_runs('/Users/{}/ETH/projects/normalization/res/fake_reference_samples/grid_b3425959/'.format(user))
    # results = collect_results_of_repetitive_runs('/Users/{}/ETH/projects/normalization/res/fake_reference_samples/grid_bb416f04/'.format(user))
