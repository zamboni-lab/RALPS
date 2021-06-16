
import os, pandas, seaborn, numpy
from src import evaluation
from matplotlib import pyplot


def choose_best_for_50_percent_NAs(best_models, g_percent, c_percent):

    try:
        # negative loss slice (desired by model construction)
        df = best_models[best_models['g_loss'] < 0]
        # grouping slice
        df = df[df['reg_grouping'] <= numpy.percentile(df['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        df['best'] = True
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def choose_best_for_2_batches(best_models, g_percent, c_percent):

    try:
        # slice by g_loss
        df = best_models[best_models['g_loss'] <= numpy.percentile(best_models['g_loss'].values, g_percent)]
        # grouping slice
        df = df[df['reg_grouping'] <= numpy.percentile(df['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        # negative loss slice (desired by model construction)

        df['best'] = True
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def plot_missing_values(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/7_batches_1.0_metabolites_{}_NAs/best_models.csv'

    all_models = pandas.DataFrame()
    for na_percent in [0.05, 0.1, 0.15, 0.2, 0.3]:
        best_models = pandas.read_csv(common_path.format(na_percent))
        best_models = best_models[best_models['best'] == True]
        best_models['NAs'] = na_percent
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='NAs', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of missing values')
    if save_path is not None:
        pyplot.savefig(save_path + 'NAs_vs_reg_grouping.pdf')

    if save_path is None:
        pyplot.show()


def plot_removed_metabolites(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/7_batches_{}_metabolites/best_models.csv'

    all_models = pandas.DataFrame()
    for percent in [0.2, 0.4, 0.6, 0.8, 1.0]:
        best_models = pandas.read_csv(common_path.format(percent))
        best_models = best_models[best_models['best'] == True]
        best_models['percent'] = percent
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='percent', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='percent', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='percent', y='b_corr', data=all_models)
    pyplot.xlabel('Percent of metabolites in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'm_percent_vs_b_corr.pdf')

    if save_path is None:
        pyplot.show()


def plot_removed_batches(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/fractions_P2_SRM_0001+P2_SPP_0001/{}_batches_1.0_metabolites/best_models.csv'

    all_models = pandas.DataFrame()
    for n in [2, 4, 7]:
        best_models = pandas.read_csv(common_path.format(n))
        if n == 2:
            best_models = choose_best_for_2_batches(best_models, 30, 70)
        else:
            best_models = best_models[best_models['best'] == True]
        best_models['n'] = n
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='n', y='reg_corr', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='reg_vc', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='b_corr', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_b_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='n', y='b_grouping', data=all_models)
    pyplot.xlabel('Number of batches in the data')
    if save_path is not None:
        pyplot.savefig(save_path + 'batches_vs_b_grouping.pdf')

    if save_path is None:
        pyplot.show()


def plot_variance_ratio(save_path=None):

    common_path = '/Users/andreidm/ETH/projects/normalization/res/variance_ratio_P2_SRM_0001_0002_0004/{}/best_models.csv'

    all_models = pandas.DataFrame()
    for ratio in [0.7, 0.8, 0.9, 0.95, 0.99]:
        best_models = pandas.read_csv(common_path.format(ratio))
        best_models = best_models[best_models['best'] == True]
        best_models['variance_ratio'] = ratio
        all_models = pandas.concat([all_models, best_models])

    seaborn.set_theme(style="whitegrid")

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_corr', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_vc', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_vc.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='reg_grouping', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_reg_grouping.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='b_corr', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_b_corr.pdf')

    pyplot.figure()
    seaborn.boxplot(x='variance_ratio', y='b_grouping', data=all_models)
    pyplot.xlabel('Percent of explained variance in PCA')
    if save_path is not None:
        pyplot.savefig(save_path + 'variance_ratio_vs_b_grouping.pdf')

    if save_path is None:
        pyplot.show()


if __name__ == "__main__":

    save_path = None

    plot_removed_batches(save_path=save_path)
    plot_removed_metabolites(save_path=save_path)
    plot_variance_ratio(save_path=save_path)
    plot_missing_values(save_path=save_path)
