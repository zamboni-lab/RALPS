
import pandas, sys, uuid, random, os, numpy, traceback, torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

from models.adversarial import run_normalization
from evaluation import evaluate_models
from constants import default_parameters_values, default_labels, default_parameters_ranges
from constants import required_config_fields
import processing


def parse_config(path=None):
    if path is not None:
        config = dict(pandas.read_csv(Path(path), index_col=0).iloc[:,0])
    else:
        config = dict(pandas.read_csv(Path(sys.argv[1]), index_col=0).iloc[:,0])

    return config


def get_data(config, parameters, n_batches=None, m_fraction=None, na_fraction=None):
    # collect data and batch info
    data = pandas.read_csv(Path(config['data_path']))
    batch_info = pandas.read_csv(Path(config['info_path']), keep_default_na=False)

    # transpose and remove annotation
    annotation = data.iloc[:, 0]
    data = data.iloc[:, 1:].T
    data.columns = annotation
    # fill in missing values
    data = data.fillna(value=parameters['min_relevant_intensity'])

    # unify ordering of samples between data and batch_info
    batch_info = batch_info.set_index(batch_info.columns[0])
    batch_info = batch_info.loc[data.index, :]

    # create prefixes for grouping
    new_index = data.index.values
    groups_indices = numpy.where(numpy.isin(batch_info['group'].astype('str'), default_labels, invert=True))[0]
    new_index[groups_indices] = 'group_' + batch_info['group'][groups_indices].astype('str') + '_' + new_index[groups_indices]

    # create prefixes for benchmarks
    benchmarks_indices = numpy.where(numpy.isin(batch_info['benchmark'].astype('str'), default_labels, invert=True))[0]
    new_index[benchmarks_indices] = 'bench_' + batch_info['benchmark'][benchmarks_indices].astype('str') + '_' + new_index[benchmarks_indices]
    data.index = new_index

    if m_fraction is not None:
        # randomly select a fraction of metabolites (for ablation experiments)
        all_metabolites = list(data.columns)
        metabolites_to_drop = random.sample(all_metabolites, int(round(1 - m_fraction, 2) * len(all_metabolites)))
        data = data.drop(labels=metabolites_to_drop, axis=1)

    if na_fraction is not None:
        # randomly mask a fraction of values (for ablation experiments)
        data = data.mask(numpy.random.random(data.shape) < na_fraction)
        data = data.fillna(parameters['min_relevant_intensity'])

    # just insert batch (mind the ordering)
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)  # shuffle

    if n_batches is not None:
        # select first n batches (for ablation experiments)
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
        return round(random.uniform(*default_parameters_ranges['d_lr']), 4)
    elif par_name == 'g_lr':
        return round(random.uniform(*default_parameters_ranges['g_lr']), 4)
    elif par_name == 'd_lambda':
        return round(random.uniform(*default_parameters_ranges['d_lambda']), 1)
    elif par_name == 'g_lambda':
        return round(random.uniform(*default_parameters_ranges['g_lambda']), 1)
    elif par_name == 'v_lambda':
        return round(random.uniform(*default_parameters_ranges['v_lambda']), 1)
    elif par_name == 'batch_size':
        return random.sample(default_parameters_ranges['batch_size'], 1)[0]
    elif par_name == 'variance_ratio':
        return random.sample(default_parameters_ranges['variance_ratio'], 1)[0]


def set_parameter(name, string_value):

    if ',' in string_value:
        # sample from options
        value = float(random.sample(string_value.split(','), 1)[0])
    elif string_value[0] != '-' and '-' in string_value:
        # sample from interval
        lower, upper = string_value.split('-')
        if '_lr' in name:
            value = round(random.uniform(float(lower), float(upper)), 4)
        elif '_lambda' in name:
            value = round(random.uniform(float(lower), float(upper)), 1)
        elif name == 'variance_ratio':
            value = round(random.uniform(float(lower), float(upper)), 2)
        else:
            value = int(random.uniform(float(lower), float(upper)))
    else:
        # set provided value
        try:
            value = float(string_value)
        except Exception:
            value = -1

        if value <= 0:
            # -1 supplied, so set defaults
            value = sample_from_default_ranges(name)

    return value


def initialise_constant_parameters(config):

    parameters = config.copy()
    # set types and defaults for constant parameters
    for int_par_name in ['latent_dim', 'n_replicates', 'epochs', 'skip_epochs', 'min_relevant_intensity']:
        try:
            parameters[int_par_name] = int(parameters[int_par_name])
            if parameters[int_par_name] <= 0:
                parameters[int_par_name] = default_parameters_values[int_par_name]
        except Exception:
            parameters[int_par_name] = default_parameters_values[int_par_name]

    for float_par_name in ['train_ratio', 'allowed_vc_increase']:
        try:
            parameters[float_par_name] = float(parameters[float_par_name])
            if parameters[float_par_name] <= 0:
                parameters[float_par_name] = default_parameters_values[float_par_name]
        except Exception:
            parameters[float_par_name] = default_parameters_values[float_par_name]

    if parameters['keep_checkpoints'].lower() not in ['true', '1']:
        parameters['keep_checkpoints'] = default_parameters_values['keep_checkpoints']
    else:
        parameters['keep_checkpoints'] = True

    if parameters['device'].lower().startswith('cuda'):
        parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        parameters['device'] = 'cpu'

    if parameters['plots_extension'].lower() not in ['png', 'pdf', 'svg']:
        parameters['plots_extension'] = default_parameters_values['plots_extension']
    else:
        parameters['plots_extension'] = parameters['plots_extension'].lower()

    return parameters


def get_pca_results(data):
    """ Run PCA on the dataset once and use fitted transformer later to set latent dims. """
    transformer = PCA()
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data.iloc[:, 1:])
    transformer.fit(scaled_data)

    return transformer


def define_latent_dim_with_pca(transformer, min_variance_ratio, n_batches):
    """ Latent_dim is defined by PCA and desired level of variance explained. """

    for i in range(0, len(transformer.explained_variance_ratio_), 5):
        # inefficient, but still fast in practice
        if sum(transformer.explained_variance_ratio_[:i]) > min_variance_ratio:
            if i < n_batches:
                return n_batches
            else:
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


def generate_parameters_grid(grid_size, parameters, data):

    parameters['n_features'] = data.shape[1]-1
    parameters['n_batches'] = data['batch'].unique().shape[0]

    # add reg_types and benchmarks to parameters
    reg_types, benchmarks = processing.extract_reg_types_and_benchmarks(data)
    parameters['reg_types'] = ','.join(reg_types)
    parameters['benchmarks'] = ','.join(benchmarks)

    if parameters['latent_dim'] <= 0:
        pca = get_pca_results(data)

    grid = []
    for _ in range(grid_size):
        new_pars = parameters.copy()
        # sample the other parameters, if not provided
        new_pars['id'] = str(uuid.uuid4())[:8]
        new_pars['d_lr'] = set_parameter('d_lr', new_pars['d_lr'])
        new_pars['g_lr'] = set_parameter('g_lr', new_pars['g_lr'])
        new_pars['d_lambda'] = set_parameter('d_lambda', new_pars['d_lambda'])
        new_pars['g_lambda'] = set_parameter('g_lambda', new_pars['g_lambda'])
        new_pars['v_lambda'] = set_parameter('v_lambda', new_pars['v_lambda'])
        new_pars['batch_size'] = int(set_parameter('batch_size', new_pars['batch_size']))
        new_pars['variance_ratio'] = set_parameter('variance_ratio', new_pars['variance_ratio'])

        if new_pars['latent_dim'] <= 0:
            # PCA itself is precomputed and reused
            new_pars['latent_dim'] = define_latent_dim_with_pca(pca, new_pars['variance_ratio'], new_pars['n_batches'])

        grid.append(new_pars)

    return grid


def check_input(config):
    """ This method makes a few sanity checks and returns True if all right. """

    message = ''
    is_correct_input = True

    # check if config contains all necessary fields
    not_all_fields_present = sum([x not in config for x in required_config_fields]) > 0
    if not_all_fields_present:
        is_correct_input = False
        message += '- Config is not complete.\n'
    else:
        # check if data file exists
        if not os.path.exists(Path(config['data_path'])):
            is_correct_input = False
            message += '- Wrong data path.\n'
        # check if batch info file exists
        elif not os.path.exists(Path(config['info_path'])):
            is_correct_input = False
            message += '- Wrong batch info path.\n'
        else:
            data = pandas.read_csv(Path(config['data_path']))
            batch_info = pandas.read_csv(Path(config['info_path']), keep_default_na=False)

            # check if names of samples match between data and batch_info files
            data_samples = sorted(list(data.iloc[:, 1:].columns))
            batch_info_samples = sorted(list(batch_info.iloc[:, 0]))
            for i in range(len(data_samples)):
                if data_samples[i] != batch_info_samples[i]:
                    is_correct_input = False
                    message += '- Samples\' names do not match between data and batch info files.\n' \
                               'Comparison of \"{}\" and \"{}\" encountered.\n'.format(data_samples[i], batch_info_samples[i])
                    break

            # check if regularization samples are provided
            reg_types = batch_info['group'].astype('str').unique().tolist()
            for value in default_labels:
                if value in reg_types:
                    reg_types.remove(value)  # default labels are not parsed as reg_types
            if len(reg_types) == 0:
                is_correct_input = False
                message += '- Regularization samples (\'group\') are missing in batch_info file.\n'

            # check if multiple batches are provided
            batch_ids = batch_info['batch'].astype('str').unique().tolist()
            for value in default_labels:
                if value in reg_types:
                    batch_ids.remove(value)  # default labels are not parsed as batch_ids
            if len(batch_ids) == 1:
                is_correct_input = False
                message += '- Only a single batch label provided: {}.\n'.format(batch_ids[0])

    return is_correct_input, message


def ralps(config):

    is_correct, warning = check_input(config)
    if is_correct:

        parameters = initialise_constant_parameters(config)
        data = get_data(config, parameters)
        grid = generate_parameters_grid(get_grid_size(config), parameters, data)

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
    ralps(config)
