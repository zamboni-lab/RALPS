
import h5py, numpy
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
from src.constants import amino_acids


if __name__ == "__main__":

    filename = '/Volumes/biol_imsb_sauer_1/users/nicola/6550_harm_4GHz/harm_4_all_short_DATA.h5'

    with h5py.File(filename, 'r') as f:

        ions_names = [str(name).replace('b\'','')[0:-1] for name in list(f["annotation"]["name"])]
        ions_mzs = [float(str(mz).replace('b\'mz','')[0:-1]) for mz in list(f["annotation"]["mzLabel"])]

        mzs = list(f["ions"]["mz"])
        data = f["data"][()].T
        cols = [str(p).replace('b\'','')[0:-1] for p in list(f["samples"]["perturbation"])]

    aa_mzs = []

    # get mz values of amino acids
    for name, _ in amino_acids:

        if name in ions_names:
            aa_index = ions_names.index(name)
            aa_mzs.append(ions_mzs[aa_index])
            # TODO: ask Nicola, why his AA mz values do not fully coincide with mine
        else:
            print(name, "is not in ions names")





    pass
