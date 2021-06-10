
import pandas, sys, uuid, random, os, numpy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from models.adversarial import run_normalization
from evaluation import evaluate_models, slice_by_grouping_and_correlation
from constants import default_parameters_values, default_labels
from constants import latent_dim_explained_variance_ratio as min_variance_ratio
from constants import grouping_threshold_percent as g_percent
from constants import correlation_threshold_percent as c_percent


def parse_config(path=None):
    if path is not None:
        config = dict(pandas.read_csv(path, index_col=0).iloc[:,0])
    else:
        config = dict(pandas.read_csv(sys.argv[1], index_col=0).iloc[:,0])

    return config


def get_data(config, n_batches=None, m_fraction=None, na_fraction=None):
    # collect data and batch info
    data = pandas.read_csv(config['data_path'])
    batch_info = pandas.read_csv(config['info_path'], keep_default_na=False)

    # transpose and remove annotation
    annotation = data.iloc[:, 0]
    data = data.iloc[:, 1:].T
    data.columns = annotation
    # fill in missing values
    data = data.fillna(config['min_relevant_intensity'])

    # create prefixes for grouping
    new_index = data.index.values
    groups_indices = numpy.where(numpy.isin(batch_info['group'].astype('str'), default_labels, invert=True))[0]
    new_index[groups_indices] = 'group_' + batch_info['group'][groups_indices].astype('str') + '_' + new_index[groups_indices]

    # create prefixes for benchmarks
    benchmarks_indices = numpy.where(numpy.isin(batch_info['group'].astype('str'), default_labels, invert=True))[0]
    new_index[benchmarks_indices] = 'bench_' + batch_info['benchmark'][benchmarks_indices].astype('str') + '_' + new_index[benchmarks_indices]
    data.index = new_index

    if m_fraction is not None:
        # randomly select a fraction of metabolites
        all_metabolites = list(data.columns)
        metabolites_to_drop = random.sample(all_metabolites, int(round(1 - m_fraction, 2) * len(all_metabolites)))
        data = data.drop(labels=metabolites_to_drop, axis=1)

    if na_fraction is not None:
        # randomly mask a fraction of values
        data = data.mask(numpy.random.random(data.shape) < na_fraction)
        data = data.fillna(config['min_relevant_intensity'])

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)

    if n_batches is not None:
        # select first n batches
        data = data.loc[data['batch'] <= n_batches, :]

    return data


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


def set_parameter(name, string_value):

    if ',' in string_value:
        # sample from options
        value = float(random.sample(string_value.split(','), 1)[0])
    elif string_value[0] != '-' and '-' in string_value:
        # sample from interval
        lower, upper = string_value.split('-')
        value = round(random.uniform(float(lower), float(upper)), 5)
    else:
        # set provided value
        value = float(string_value)
        if value <= 0:
            # -1 supplied, so set defaults
            value = sample_from_default_ranges(name)

    return value


def initialise_constant_parameters(config):

    parameters = config.copy()
    # set types and defaults for constant parameters
    for int_par_name in ['latent_dim', 'n_replicates', 'epochs', 'skip_epochs', 'callback_step', 'min_relevant_intensity']:
        try:
            parameters[int_par_name] = int(parameters[int_par_name])
            if parameters[int_par_name] < 0:
                parameters[int_par_name] = default_parameters_values[int_par_name]
        except Exception:
            parameters[int_par_name] = default_parameters_values[int_par_name]

    try:
        parameters['train_ratio'] = float(parameters['train_ratio'])
        if parameters['train_ratio'] <= 0:
            parameters['train_ratio'] = default_parameters_values['train_ratio']
    except Exception:
        parameters['train_ratio'] = default_parameters_values['train_ratio']

    if parameters['keep_checkpoints'].lower() not in ['true', '1']:
        parameters['keep_checkpoints'] = default_parameters_values['keep_checkpoints']
    else:
        parameters['keep_checkpoints'] = True

    return parameters


def define_latent_dim_with_pca(data):
    """ Latent_dim is defined by PCA and desired level of variance explained (defaults to 0.99). """

    transformer = PCA()
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    transformer.fit(scaled_data)

    for i in range(0, len(transformer.explained_variance_ratio_), 5):
        if sum(transformer.explained_variance_ratio_[:i]) > min_variance_ratio:
            return i


def extract_reg_types_and_benchmarks(data):
    """ This method extracts the names for regularization sample types and benchmark samples from the data. """
    reg_types = set()
    benchmarks = set()
    # parse reg_types and benchmarks from data
    for i in range(data.index.shape[0]):
        if 'group_' in data.index[i]:
            reg_types.add('group_{}'.format(data.index[i].split('group')[1].split('_')[1]))
        if 'bench_' in data.index[i]:
            benchmarks.add('bench_{}'.format(data.index[i].split('bench')[1].split('_')[1]))

    return list(reg_types), list(benchmarks)


def generate_parameters_grid(config, data):

    parameters = initialise_constant_parameters(config)
    parameters['n_features'] = data.shape[1]-1
    parameters['n_batches'] = data['batch'].unique().shape[0]

    if parameters['latent_dim'] <= 0:
        parameters['latent_dim'] = define_latent_dim_with_pca(data)

    # add reg_types and benchmarks to parameters
    reg_types, benchmarks = extract_reg_types_and_benchmarks(data)
    parameters['reg_types'] = ','.join(reg_types)
    parameters['benchmarks'] = ','.join(benchmarks)

    grid = []
    for _ in range(get_grid_size(config)):
        new_pars = parameters.copy()
        # sample the other parameters, if not provided
        new_pars['id'] = str(uuid.uuid4())[:8]
        new_pars['d_lr'] = set_parameter('d_lr', new_pars['d_lr'])
        new_pars['g_lr'] = set_parameter('g_lr', new_pars['g_lr'])
        new_pars['d_lambda'] = set_parameter('d_lambda', new_pars['d_lambda'])
        new_pars['g_lambda'] = set_parameter('g_lambda', new_pars['g_lambda'])
        new_pars['batch_size'] = int(set_parameter('batch_size', new_pars['batch_size']))

        grid.append(new_pars)

    return grid


def harmae(config):

    data = get_data(config)
    grid = generate_parameters_grid(config, data)

    for parameters in tqdm(grid):
        run_normalization(data, parameters)

    evaluate_models(config)


if __name__ == "__main__":

    # config = parse_config(sys.argv[1])
    config = parse_config('/Users/andreidm/ETH/projects/normalization/data/sarah/config.csv')
    harmae(config)
