
import h5py, numpy, pandas
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
from src.constants import amino_acids, allowed_ppm_error
from src.constants import shared_perturbations as sps
from src.constants import batches as bids
from matplotlib import pyplot
from src.utils.combat import combat
from src.preprocessing import get_all_data_from_h5


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


def get_tic_normalized_amino_acid_values(data, colnames, aa_intensities):
    """ This method does simplistic normalization to compare intensities of amino acids
        of batches of different experiments. For each AA, intensities are divided by the TIC
        of the corresponding experiment. """

    experiment_ids = list(set(colnames))

    normalized_intensities = []
    for i in range(len(aa_intensities)):
        experiments = []
        for j in range(len(experiment_ids)):
            # get indices of current experiment in data (its columns)
            j_experiment_indices = numpy.where(numpy.array(colnames) == experiment_ids[j])[0]
            # normalize by TIC, i.e. by corresponding column in data
            aa_values = aa_intensities[i][j] / numpy.sum(data[:, j_experiment_indices])  # * scaling_factor
            experiments.append(aa_values)
        normalized_intensities.append(experiments)

    return normalized_intensities


def get_combat_normalized_amino_acid_values(data, colnames, aa_mz_indices):
    """ This method performs ComBat normalization to compare intensities of amino acids
        of batches of different experiments. """

    # try combat normalisation
    data = pandas.DataFrame(data)
    unique_cols = [colnames[i] + '#' + str(i) for i in range(len(colnames))]  # make colnames unique
    data.columns = unique_cols

    # assign batches
    flatten = lambda l: [item for sublist in l for item in sublist]
    batches = flatten([[i, i, i] for i in range(len(colnames) // 3)])

    batches = pandas.DataFrame(batches)
    batches.columns = ["batch"]
    batches.index = unique_cols

    normalized_data = combat(data, batches["batch"], None)

    normalized_intensities = get_intensities_data_structure(numpy.array(normalized_data), colnames, aa_mz_indices)

    return normalized_intensities


def get_amino_acids_indices_in_data(ions_mzs, ions_names, mz_axis):
    """ This method locates amino acids by name in the list of detected ions peaks.
        It finds the closest peak on the spectrum to the aligned mz of each amino acid
        and returns a list of mz indices and its names. """

    aa_mz_indices = []  # indices of amino acid mz values on the mz_axis
    aa_names = []

    # get mz values of amino acids
    for name, _ in amino_acids:
        # if this amino acid appears on the spectra
        if name in ions_names:
            aa_index = ions_names.index(name)
            aa_mz_indices.append(find_closest_ion_mz_index(mz_axis, ions_mzs[aa_index]))
            aa_names.append(name)
            # TODO: ask Nicola, why his AA mz values do not fully coincide with mine
        else:
            print(name, "is not in ions names")

    return aa_mz_indices, aa_names


def get_intensities_data_structure(data, colnames, aa_mz_indices):
    """ This method forms and returns a convenient data structure to work with. """

    experiment_ids = list(set(colnames))

    # now create a convenient data structure with all intensities of amino acids
    aa_intensities = []
    for mz_index in aa_mz_indices:
        experiments = []
        for id in experiment_ids:
            experiments.append(data[mz_index, numpy.where(numpy.array(colnames) == id)[0]])
        aa_intensities.append(experiments)

    return aa_intensities


def compare_two_normalisations_for_aa():

    filename = '/Users/dmitrav/ETH/projects/normalization/data/raw/harm_4_all_short_DATA.h5'
    all_data = get_all_data_from_h5(filename)

    aa_mz_indices, aa_names = get_amino_acids_indices_in_data(all_data["annotation"]["mzs"],
                                                              all_data["annotation"]["names"],
                                                              all_data["samples"]["mzs"])
    data = all_data["samples"]["data"]
    aa_intensities = get_intensities_data_structure(data, all_data["samples"]["names"], aa_mz_indices)

    # try tic normalisation
    tic_normalized = get_tic_normalized_amino_acid_values(data, all_data["samples"]["names"], aa_intensities)

    # try combat normalisation
    combat_normalized = get_combat_normalized_amino_acid_values(data, all_data["samples"]["names"], aa_mz_indices)

    plot_amino_acids_over_experiments(aa_intensities)  # raw data
    plot_amino_acids_over_experiments(tic_normalized)  # shows that tic normalization doesn't work properly
    plot_amino_acids_over_experiments(combat_normalized)  # combat normalisation


if __name__ == "__main__":

    # batch effects and simple normalisations for the first committee meeting
    compare_two_normalisations_for_aa()

