
import numpy, pandas, seaborn
from matplotlib import pyplot


def get_samples_by_types_dict(samples_names, types_of_interest):
    """ Create a dict like this: {'P1_FA_0001': ['P1_FA_0001_0306_0', ..., 'P1_FA_0001_0306_2'], ...}  """

    if types_of_interest is None:
        # get unique types + filter out diluted samples
        types_of_interest = list(set(['_'.join(x.split('_')[:3]) for x in samples_names if x.split('_')[2] == '0001']))

    samples_by_types = {}
    for i, sample in enumerate(samples_names):
        # check which type this sample has
        for type in types_of_interest:
            if type in sample and type not in samples_by_types:
                # if new type, putt in the dict, create a list
                samples_by_types[type] = [sample]
                break
            elif type in sample and type in samples_by_types:
                # if type already exists in the dict, append sample
                samples_by_types[type].append(sample)
                break
            else:
                pass

    return samples_by_types


def plot_batch_cross_correlations(data, method_name, sample_types_of_interest=None, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    samples_by_types = get_samples_by_types_dict(data.columns.values, sample_types_of_interest)

    if sample_types_of_interest is None:
        for i, type in enumerate(samples_by_types):
            df = data.loc[:, numpy.array(samples_by_types[type])]
            df.columns = [x[-6:] for x in df.columns]
            df = df.corr()

            seaborn.heatmap(df)
            pyplot.title('Cross correlations: {}: {}'.format(type, method_name))
            pyplot.tight_layout()
            pyplot.show()

    else:
        pyplot.figure(figsize=(12,8))

        for i, type in enumerate(samples_by_types):
            df = data.loc[:, numpy.array(samples_by_types[type])]
            df.columns = [x[-6:] for x in df.columns]
            df = df.corr()

            ax = pyplot.subplot(2, 3, i+1)
            seaborn.heatmap(df)
            ax.set_title(type)

        pyplot.suptitle('Cross correlations: {}'.format(method_name))
        pyplot.tight_layout()
        # pyplot.show()
        pyplot.savefig(save_to + 'correlations_{}.pdf'.format(method_name.replace(' ', '_')))


def compute_cv_for_samples_types(data, sample_types_of_interest=None):

    samples_by_types = get_samples_by_types_dict(data.columns.values, sample_types_of_interest)

    cv_dict = {}
    for i, type in enumerate(samples_by_types):
        values = data.loc[:, numpy.array(samples_by_types[type])].values
        values = values.flatten()
        cv_dict[type] = numpy.std(values) / numpy.mean(values)

    return cv_dict


if __name__ == '__main__':

    data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data.csv')
    data = data.iloc[:, 3:]

    # plot_batch_cross_correlations(data, 'original samples',
    #                               sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
    #                                                         'P2_SFA_0001', 'P2_SRM_0001',
    #                                                         'P2_SFA_0002', 'P1_FA_0008'])

    res = compute_cv_for_samples_types(data, sample_types_of_interest=['P1_FA_0001', 'P2_SF_0001',
                                                                       'P2_SFA_0001', 'P2_SRM_0001',
                                                                       'P2_SFA_0002', 'P1_FA_0008'])

    print(res)