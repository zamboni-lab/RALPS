
import numpy, pandas

if __name__ == '__main__':

    f_data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/all_data.csv')
    f_data_ori = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/running_on_all/Ori.csv')
    f_data_nobe = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/res/running_on_all/Rec_nobe.csv')

    print()
