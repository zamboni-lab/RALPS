
import pandas, sys, uuid, random
from tqdm import tqdm

from src.models.adversarial import get_data, run_normalization
from src.constants import default_parameters_values


def parse_config(path=None):
    if path is not None:
        config = dict(pandas.read_csv(path, index_col=0).iloc[:,0])
    else:
        config = dict(pandas.read_csv(sys.argv[1], index_col=0).iloc[:,0])

    return config


def get_grid_size(config):

    try:
        grid_size = int(config['grid_size'])
        if grid_size <= 0:
            return default_parameters_values['grid_size']
    except Exception:
        return default_parameters_values['grid_size']

    return grid_size


def sample_from_default_ranges(par_name):

    if par_name == 'd_lr':
        return round(random.uniform(0.00005, 0.005), 5)
    elif par_name == 'g_lr':
        return round(random.uniform(0.00005, 0.005), 5)
    elif par_name == 'd_lambda':
        return round(random.uniform(0., 10.), 1)
    elif par_name == 'g_lambda':
        return round(random.uniform(0., 10.), 1)
    elif par_name == 'batch_size':
        return random.sample([32, 64, 128], 1)[0]


def sample_parameter(name, string_value):

    if ',' in string_value:
        value = float(random.sample(string_value.split(','), 1)[0])
    elif string_value[0] != '-' and '-' in string_value:
        lower, upper = string_value.split('-')  # interval
        value = round(random.uniform(float(lower), float(upper)), 5)
    else:
        value = float(string_value)
        if value < 0:
            # -1 supplied, so
            value = sample_from_default_ranges(name)

    return value


def initialise_constant_parameters(config):

    parameters = config.copy()
    # set types and defaults for constant parameters
    for int_par_name in ['latent_dim', 'n_replicates', 'epochs', 'skip_epochs', 'callback_step']:
        try:
            parameters[int_par_name] = int(parameters[int_par_name])
        except Exception:
            parameters[int_par_name] = default_parameters_values[int_par_name]
    try:
        parameters['train_ratio'] = float(parameters['train_ratio'])
    except Exception:
        parameters['train_ratio'] = default_parameters_values['train_ratio']
    try:
        parameters['keep_checkpoints'] = bool(parameters['keep_checkpoints'])
    except Exception:
        parameters['keep_checkpoints'] = default_parameters_values['keep_checkpoints']

    return parameters


def generate_parameters_grid(config, data):

    parameters = initialise_constant_parameters(config)
    parameters['n_features'] = data.shape[1]-1
    parameters['n_batches'] = data['batch'].unique().shape[0]

    reg_types = set()
    benchmarks = set()
    # parse reg_types and benchmarks from data
    for i in range(data.index.shape[0]):
        if 'group_' in data.index[i]:
            reg_types.add('group_{}'.format(data.index[i].split('group')[1].split('_')[1]))
        if 'bench_' in data.index[i]:
            benchmarks.add('bench_{}'.format(data.index[i].split('bench')[1].split('_')[1]))
    # add to parameters
    parameters['reg_types'] = ','.join(list(reg_types))
    parameters['benchmarks'] = ','.join(list(benchmarks))

    grid = []
    for _ in range(get_grid_size(config)):
        new_set = parameters.copy()
        # sample the other parameters
        new_set['id'] = str(uuid.uuid4())[:8]
        new_set['d_lr'] = sample_parameter('d_lr', new_set['d_lr'])
        new_set['g_lr'] = sample_parameter('g_lr', new_set['g_lr'])
        new_set['d_lambda'] = sample_parameter('d_lambda', new_set['d_lambda'])
        new_set['g_lambda'] = sample_parameter('g_lambda', new_set['g_lambda'])
        new_set['batch_size'] = int(sample_parameter('batch_size', new_set['batch_size']))

        grid.append(new_set)

    return grid


if __name__ == "__main__":

    # read config file
    config = parse_config(path='/Users/andreidm/ETH/projects/normalization/data/config_v4.csv')

    data = get_data(config['data_path'], config['info_path'])
    grid = generate_parameters_grid(config, data)

    for parameters in tqdm(grid):
        # run grid
        run_normalization(data, parameters)

    # print best epochs


    pass