
import h5py, numpy
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
from src.constants import amino_acids, allowed_ppm_error
from matplotlib import pyplot


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


def plot_amino_acids_over_experiments(aa_intensities):
    """ This method helps to assess reproducibility of the spectra of amino acids.
        It plots intensities of amino acids over different experiments. """

    x = [x for x in range(len(aa_intensities[0][0]))]

    for i in range(len(aa_intensities)):
        fig, axs = pyplot.subplots(4, figsize=(8, 8))

        axs[0].plot(aa_intensities[i][0], 'o-')
        axs[0].set_title(aa_names[i])
        axs[0].xaxis.set_ticks(x)
        axs[0].grid()

        axs[1].plot(aa_intensities[i][1], 'o-')
        axs[1].xaxis.set_ticks(x)
        axs[1].grid()

        axs[2].plot(aa_intensities[i][2], 'o-')
        axs[2].xaxis.set_ticks(x)
        axs[2].grid()

        axs[3].plot(aa_intensities[i][3], 'o-')
        axs[3].xaxis.set_ticks(x)
        axs[3].grid()

        pyplot.tight_layout()
        pyplot.show()



if __name__ == "__main__":

    filename = '/Volumes/biol_imsb_sauer_1/users/nicola/6550_harm_4GHz/harm_4_all_short_DATA.h5'

    with h5py.File(filename, 'r') as f:

        ions_names = [str(name).replace('b\'','')[0:-1] for name in list(f["annotation"]["name"])]
        ions_mzs = [float(str(mz).replace('b\'mz','')[0:-1]) for mz in list(f["annotation"]["mzLabel"])]

        mz_axis = list(f["ions"]["mz"])
        data = f["data"][()].T
        cols = [str(p).replace('b\'','')[0:-1] for p in list(f["samples"]["perturbation"])]
        experiment_ids = list(set(cols))

    aa_mzs_indices = []  # indices of amino acid mz values on the mz_axis
    aa_names = []

    # get mz values of amino acids
    for name, _ in amino_acids:
        # if this amino acid appears on the spectra
        if name in ions_names:
            aa_index = ions_names.index(name)
            aa_mzs_indices.append(find_closest_ion_mz_index(mz_axis, ions_mzs[aa_index]))
            aa_names.append(name)
            # TODO: ask Nicola, why his AA mz values do not fully coincide with mine
        else:
            print(name, "is not in ions names")

    # now create a convenient data structure with all intensities of amino acids
    aa_intensities = []
    for mz_index in aa_mzs_indices:
        experiments = []
        for id in experiment_ids:
            experiments.append(data[mz_index, numpy.where(numpy.array(cols) == id)[0]])
        aa_intensities.append(experiments)

    # TODO: try simplest normalisation strategies to reproduce intensities better

    if True:
        plot_amino_acids_over_experiments(aa_intensities)

    pass
