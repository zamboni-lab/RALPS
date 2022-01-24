
import numpy


def get_initial_samples_names(data_index):
    """ This method removes auxiliary prefixes from the names of regularization and benchmark samples. """

    initial_names = []
    for name in list(data_index):
        if 'bench_' in name and 'group_' in name:
            initial_names.append('_'.join(name.split('_')[4:]))
        elif 'bench_' in name or 'group_' in name:
            initial_names.append('_'.join(name.split('_')[2:]))
        else:
            initial_names.append(name)

    return initial_names


def split_to_train_and_test(values, batches, scaler, proportion=0.7):
    """ Split data for the classifier of the adversarial training loop. """

    n_samples, n_features = values.shape

    # scale
    scaled = scaler.transform(values)
    # split values to train and val
    x_train = scaled[:int(proportion * n_samples), :]
    x_val = scaled[int(proportion * n_samples):, :]
    y_train = batches[:int(proportion * n_samples)]
    y_val = batches[int(proportion * n_samples):]

    if numpy.min(batches) == 1:
        # enumerate batches from 0 to n
        y_train -= 1
        y_val -= 1

    return x_train, x_val, y_train, y_val


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
