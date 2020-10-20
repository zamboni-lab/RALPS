
import numpy, pandas, scipy, seaborn, math, time, umap, h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.constants import batches as bids
from src.constants import shared_perturbations as sps



def run_pca(data):

    transformer = PCA(n_components=100)
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    reduced_data = transformer.fit_transform(scaled_data)

    print(reduced_data.shape)

    # percent of variance explained
    print(transformer.explained_variance_ratio_ * 100)

    return reduced_data


def get_all_data_from_h5(path):
    """ This method parses h5 file to extract all necessary data. """

    with h5py.File(path, 'r') as f:

        ions_names = [str(name).replace('b\'', '')[0:-1] for name in list(f["annotation"]["name"])]
        ions_mzs = [float(str(mz).replace('b\'mz', '')[0:-1]) for mz in list(f["annotation"]["mzLabel"])]

        mz_axis = list(f["ions"]["mz"])
        data = f["data"][()].T
        colnames = [str(p).replace('b\'', '')[0:-1] for p in list(f["samples"]["perturbation"])]

    all_data = {
        "annotation": {"mzs": ions_mzs, "names": ions_names},
        "samples": {"data": data, "mzs": mz_axis, "names": colnames}
    }

    return all_data


def check_shared_perturbations():
    """ Check how many perturbations are done in each batch and how many are shared. """

    path = '/Users/andreidm/ETH/projects/normalization/data/'

    perturbations = []
    unique_perturbations = set()

    for name in ['harm_4_0108 DATA.h5', 'harm_4_0110 DATA.h5', 'harm_4_0124 DATA.h5',
                 'harm_4_0219 DATA.h5', 'harm_4_0221 DATA.h5', 'harm_4_0304 DATA.h5',
                 'harm_4_0306 DATA.h5']:

        data = get_all_data_from_h5(path + name)

        print()
        print('processing {}'.format(name))
        print('n perturbations = {}'.format(set(data['samples']['names']).__len__()))

        perturbations.append(list(set(data['samples']['names'])))
        unique_perturbations.update(list(set(data['samples']['names'])))

    common_perturbations = []
    for up in list(unique_perturbations):
        # make sure it exists in all batches
        exists_in_all_batches = sum([up in batch for batch in perturbations]) == 7
        if exists_in_all_batches:
            common_perturbations.append(up)

    print("total shared perturbations for 7 batches: {}".format(len(common_perturbations)))
    print(common_perturbations)


def get_shared_perturbations_ids_for_batch(batch):

    ids = []
    names = []

    for sp in sps:
        # get ids of shared perturbations in this batch
        sp_ids = numpy.where(numpy.array(batch['data']['samples']['names']) == sp)[0].tolist()

        ids.extend(sp_ids)  # add ids
        names.extend([sp + '_' + batch['id'] + '_' + str(i) for i in range(len(sp_ids))])  # add names (perturbation + batch + replicate)

    return ids, names


def merge_batches_and_save_dataset():
    """ It gets all batches, merges mz axis
    and makes a single dataset of shared perturbations (samples with spike-ins). """

    path = '/Users/andreidm/ETH/projects/normalization/data/'

    batches = []
    merged_mz = set()

    for bid in bids:

        data = get_all_data_from_h5(path + 'harm_4_{}_DATA.h5'.format(bid))
        batches.append({'data': data, 'id': bid})
        merged_mz.update(data['samples']['mzs'])

    merged_mz = sorted(list(merged_mz))

    all_data = pandas.DataFrame({'name': ['' for x in merged_mz], 'mz': merged_mz, 'rt': [1 for x in merged_mz]})

    shared_mz_df = pandas.DataFrame()
    for mz in merged_mz:

        mz_df = pandas.DataFrame()
        for batch in batches:

            columns, names = get_shared_perturbations_ids_for_batch(batch)

            if mz in batch['data']['samples']['mzs']:
                # if this mz appears in batch, use intensities
                index = batch['data']['samples']['mzs'].index(mz)
                bdf = pandas.DataFrame([batch['data']['samples']['data'][index, columns]], columns=names)
            else:
                # if not, fill up with zeros
                bdf = pandas.DataFrame([numpy.zeros(len(columns))], columns=names)

            mz_df = pandas.concat([mz_df, bdf], axis=1)

        shared_mz_df = pandas.concat([shared_mz_df, mz_df], ignore_index=True)

    all_data = pandas.concat([all_data, shared_mz_df], axis=1)
    all_data.to_csv(path + "all_data.csv", index=False)


if __name__ == '__main__':
    """
    # check which samples / perturbations are common for each batch
    check_shared_perturbations()
    """


    pass