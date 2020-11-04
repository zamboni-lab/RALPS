
import pandas, numpy, uuid, os, random, sys
from multiprocessing import Process, Pool
from matplotlib import pyplot
from src.models import adversarial


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


def generate_grid(g_loss, regularization, grid_size, grid_name, save_to):

    grid = range(grid_size)

    parameters = {

        'in_path': ['/Users/andreidm/ETH/projects/normalization/data/' for x in grid],
        'out_path': ['/Users/andreidm/ETH/projects/normalization/res/grid_search/' for x in grid],
        'id': [str(uuid.uuid4())[:12] for x in grid],

        'n_features': [170 for x in grid],  # n of metabolites in initial dataset
        'latent_dim': [100 for x in grid],  # n dimensions to reduce to
        'n_batches': [7 for x in grid],

        'd_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # discriminator learning rate
        'g_lr': [round(random.uniform(5e-5, 5e-3), 4) for x in grid],  # generator learning rate
        'd_loss': ['CE' for x in grid],
        'g_loss': [g_loss for x in grid],
        'd_lambda': [round(random.uniform(0.1, 5), 1) for x in grid],  # discriminator regularization term coefficient
        'g_lambda':  [round(random.uniform(0.1, 5), 1) for x in grid],  # generator regularization term coefficient
        'use_g_regularization': [regularization for x in grid],  # whether to use generator regularization term
        'train_ratio': [0.7 for x in grid],  # for train-test split
        'batch_size': [64 for x in grid],
        'g_epochs': [0 for x in grid],  # pretraining of generator
        'd_epochs': [0 for x in grid],  # pretraining of discriminator
        'adversarial_epochs': [200 for x in grid],  # simultaneous competitive training

        'callback_step': [-1 for x in grid],  # save callbacks every n epochs
        'keep_checkpoints': [False for x in grid]  # whether to keep all checkpoints, or just the best epoch
    }

    grid = pandas.DataFrame(parameters)
    grid.to_csv(save_to + 'grid_{}.csv'.format(grid_name))
    print('grid {} saved'.format(grid_name))


def generate_grids():

    save_to = '/Users/andreidm/ETH/projects/normalization/data/'
    generate_grid('MSE', True, 100, 'mse_reg', save_to)
    generate_grid('MSE', False, 100, 'mse', save_to)
    generate_grid('L1', True, 100, 'l1_reg', save_to)
    generate_grid('L1', False, 100, 'l1', save_to)
    generate_grid('SL1', True, 100, 'sl1_reg', save_to)
    generate_grid('SL1', False, 100, 'sl1', save_to)


if __name__ == "__main__":

    name = sys.argv[1]

    path = '/Users/andreidm/ETH/projects/normalization/data/'
    grid = pandas.read_csv(path + name, index_col=0)

    for i in range(grid.shape[0]):
        parameters = dict(grid.iloc[i, :])
        adversarial.main(parameters)


