
import h5py, numpy
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
from src.constants import amino_acids, allowed_ppm_error


def find_closest_ion_mz_index(mz_axis, ion_mz):
    """ This method find the closest mz on the whole mz axis for the specified ion mz. """

    closest_index = 0
    while mz_axis[closest_index] < ion_mz and closest_index+1 < len(mz_axis):
        closest_index += 1

    previous_peak_ppm = abs(mz_axis[closest_index-1] - ion_mz) / ion_mz * 10 ** 6
    next_peak_ppm = abs(mz_axis[closest_index] - ion_mz) / ion_mz * 10 ** 6

    if previous_peak_ppm <= next_peak_ppm and previous_peak_ppm <= allowed_ppm_error:
        return closest_index-1

    elif previous_peak_ppm > next_peak_ppm and next_peak_ppm <= allowed_ppm_error:
        return closest_index

    else:
        print("ppms:", previous_peak_ppm, next_peak_ppm)
        raise ValueError("No ion within current ppm identified!")


if __name__ == "__main__":

    filename = '/Volumes/biol_imsb_sauer_1/users/nicola/6550_harm_4GHz/harm_4_all_short_DATA.h5'

    with h5py.File(filename, 'r') as f:

        ions_names = [str(name).replace('b\'','')[0:-1] for name in list(f["annotation"]["name"])]
        ions_mzs = [float(str(mz).replace('b\'mz','')[0:-1]) for mz in list(f["annotation"]["mzLabel"])]

        mzs = list(f["ions"]["mz"])
        data = f["data"][()].T
        cols = [str(p).replace('b\'','')[0:-1] for p in list(f["samples"]["perturbation"])]

    aa_mzs_indices = []
    aa_names = []

    # get mz values of amino acids
    for name, _ in amino_acids:

        if name in ions_names:
            aa_index = ions_names.index(name)
            aa_mzs_indices.append(find_closest_ion_mz_index(mzs, ions_mzs[aa_index]))
            aa_names.append(name)
            # TODO: ask Nicola, why his AA mz values do not fully coincide with mine
        else:
            print(name, "is not in ions names")

    aa_intensities = []
    for mz_index in aa_mzs_indices:
        experiments = []
        for experiment in list(set(cols)):
            experiments.append(data[mz_index, numpy.where(numpy.array(cols) == experiment)[0]])
        aa_intensities.append(experiments)

    # TODO: plot intensities now and figure out why they differ
    pass
