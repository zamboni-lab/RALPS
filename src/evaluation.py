
import seaborn, pandas, numpy, os
from matplotlib import pyplot
from pathlib import Path

from constants import grouping_threshold_percent as g_percent
from constants import correlation_threshold_percent as c_percent


def mask_non_relevant_intensities(reconstruction, min_relevant_intensity):
    """ This method replaces intensities below threshold with zeros. """
    values = reconstruction.values
    values[values < min_relevant_intensity] = 0
    return pandas.DataFrame(values, index=reconstruction.index, columns=reconstruction.columns)


def evaluate_models(config):
    """ This method evaluates results of the grid search.
        If found, best models are printed. All models' metrics are logged to a file. """

    out_path = Path(config['out_path'])

    best_models = pandas.DataFrame()
    for id in os.listdir(out_path):
        if id.startswith('.'):
            pass
        else:
            results_path = out_path / id / 'history_{}.csv'.format(id)
            pars_path = out_path / id / 'parameters_{}.csv'.format(id)

            if os.path.exists(results_path) and os.path.exists(pars_path):
                id_results = pandas.read_csv(results_path)
                id_results = id_results.loc[id_results['solution'] == True, :]
                if id_results.shape[0] > 0:
                    stopped_early = pandas.read_csv(pars_path, index_col=0).T
                    stopped_early = str(stopped_early['stopped_early'].values[0])
                    id_results['id'] = id
                    id_results['stopped_early'] = stopped_early
                    best_models = pandas.concat([best_models, id_results])
                else:
                    pass
                del id_results

    ids = best_models['id']
    best_models = best_models.drop(columns=['solution', 'id'])
    best_models.insert(1, 'best', False)
    best_models.insert(2, 'id', ids)
    # pick best models
    print('GRID SEARCH BEST MODELS:', '\n')
    if best_models.shape[0] == 0:
        print('WARNING: no solutions found! Check input data, try other parameters, or report an issue.\n')
    else:
        top = select_top_solutions(best_models, g_percent, c_percent)
        if top is None:
                print('WARNING: could not find the best solution, returning the full list sorted by reg_grouping\n')
                best_models = best_models.sort_values('reg_grouping')
                print(best_models.to_string(), '\n')
        else:
            for top_id in top['id'].values:
                # mark the best models
                best_models.loc[best_models['id'] == top_id, 'best'] = True
            print(top.to_string(), '\n')

        best_models.to_csv(out_path / 'best_models.csv', index=False)
        print('full grid saved to {}'.format(out_path / 'best_models.csv'))


def slice_by_grouping_and_correlation(history, g_percent, c_percent):
    """ This method is used to select the best epoch of a single solution.
        It applies thresholds and sorting for the three key quality metrics. """

    try:
        # grouping slice
        df = history[history['reg_grouping'] <= numpy.percentile(history['reg_grouping'].values, g_percent)]
        # correlation slice + sorting by reconstruction loss
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('rec_loss')
        assert df.shape[0] > 0
    except Exception:
        df = None
    return df


def select_top_solutions(history, g_percent, c_percent):
    """ This method is used to select the best solutions in the grid.
        It follows similar logic as in epoch selection, but accounts for benchmarks as well. """

    if (history['b_corr'] >= 0).sum() > 0 and (history['b_grouping'] >= 0).sum() > 0:
        # there are benchmarks to account for
        history['all_corr'] = history['reg_corr'] + history['b_corr']
        history['all_grouping'] = history['reg_grouping'] + history['b_grouping']
    else:
        history['all_corr'] = history['reg_corr']
        history['all_grouping'] = history['reg_grouping']

    df = pandas.DataFrame()
    try:
        df = pandas.concat([df, history[
            (history['batch_vc'] <= numpy.percentile(history['batch_vc'].values, g_percent))
            & (history['all_corr'] >= numpy.percentile(history['all_corr'].values, c_percent))
            ]])
        df = pandas.concat([df, history[
            (history['all_grouping'] <= numpy.percentile(history['all_grouping'].values, g_percent))
            & (history['all_corr'] >= numpy.percentile(history['all_corr'].values, c_percent))
            ]])
        df = df.drop_duplicates().sort_values('rec_loss')

        assert df.shape[0] > 0
        df['best'] = True
    except Exception:
        df = None
    return df


def find_best_epoch(history, skip_epochs, mean_batch_vc_original, mean_reg_vc_original):
    """ This method seeks for the best epoch using logged history of quality metrics. """

    # skip first n epochs
    if 0 < skip_epochs < history.shape[0]:
        history = history.iloc[skip_epochs:, :]

    # filter out epochs of increased variation coefs
    history = history.loc[(history['batch_vc'] < mean_batch_vc_original) & (history['reg_vc'] < mean_reg_vc_original), :]
    if history.shape[0] < 1:
        print('WARNING: increased VCs -> no solution for current parameter set')
        return None
    else:
        df = slice_by_grouping_and_correlation(history, 10, 90)
        if df is None:
            df = slice_by_grouping_and_correlation(history, 20, 80)
            if df is None:
                df = slice_by_grouping_and_correlation(history, 30, 70)
                if df is None:
                    df = slice_by_grouping_and_correlation(history, 40, 60)
                    if df is None:
                        df = slice_by_grouping_and_correlation(history, 50, 50)
                        if df is None:
                            min_grouping_epochs = history.loc[history['reg_grouping'] == history['reg_grouping'].min(), :]
                            if min_grouping_epochs.shape[0] > 1:
                                # min grouping + max correlation
                                best_epoch = int(min_grouping_epochs.loc[min_grouping_epochs['reg_corr'] == min_grouping_epochs['reg_corr'].max(), 'epoch'].values[-1])
                                print('WARNING: couldn\'t find the best epoch, '
                                      'returning the one of min grouping with max cross-correlation: epoch {}'.format(best_epoch))
                            else:
                                # min grouping
                                best_epoch = int(history.loc[history['reg_grouping'] == history['reg_grouping'].min(), 'epoch'].values[-1])
                                print('WARNING: couldn\'t find the best epoch, '
                                      'returning the last one of min grouping coef: epoch {}'.format(best_epoch))

                            return best_epoch

        return int(df['epoch'].values[0])


def plot_losses(rec_loss, d_loss, g_loss, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(3, figsize=(6,9))

    fig.suptitle('Adversarial training loop losses')

    axs[0].plot(range(1, 1+len(d_loss)), d_loss)
    axs[0].axvline(best_epoch, c='black', label='Best')
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Cross-entropy')
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(g_loss)), g_loss)
    axs[1].axvline(best_epoch, c='black', label='Best')
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Regularized MSE - Cross-entropy')
    axs[1].grid(True)

    axs[2].plot(range(1, 1 + len(rec_loss)), rec_loss)
    axs[2].axvline(best_epoch, c='black', label='Best')
    axs[2].set_title('Reconstruction loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MSE')
    axs[2].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to / 'losses_{}.{}'.format(parameters['id'], parameters['plots_extension']))
    pyplot.close()


def plot_metrics(d_accuracy, reg_correlation, reg_clustering, reg_vc, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(2, 2, figsize=(9,6))

    fig.suptitle('Adversarial training loop metrics')

    axs[0,0].plot(range(1, 1+len(d_accuracy)), d_accuracy)
    axs[0,0].axvline(best_epoch, c='black', label='Best')
    axs[0,0].set_title('Validation classifier accuracy')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Accuracy')
    axs[0,0].grid(True)

    axs[1,0].plot(range(1, 1 + len(reg_correlation)), reg_correlation)
    axs[1,0].axvline(best_epoch, c='black', label='Best')
    axs[1,0].set_title('Mean samples\' cross-correlation estimates')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Pearson coef')
    axs[1,0].grid(True)

    axs[0,1].plot(range(1, 1 + len(reg_clustering)), reg_clustering)
    axs[0,1].axvline(best_epoch, c='black', label='Best')
    axs[0,1].set_title('Mean estimate of samples\' grouping')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('HBDSCAN distance')
    axs[0,1].grid(True)

    axs[1,1].plot(range(1, 1+len(reg_vc)), reg_vc)
    axs[1,1].axvline(best_epoch, c='black', label='Best')
    axs[1,1].set_title('Mean samples\' variation coef')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('VC')
    axs[1,1].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to / 'metrics_{}.{}'.format(parameters['id'], parameters['plots_extension']))
    pyplot.close()


def plot_benchmarks_metrics(b_correlations, b_grouping, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(2, figsize=(6,6))

    fig.suptitle('Benchmarks metrics')

    axs[0].plot(range(1, 1+len(b_correlations)), b_correlations)
    axs[0].axvline(best_epoch, c='black', label='Best')
    axs[0].set_title('Mean benchmark cross-correlation')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Pearson coef')
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(b_grouping)), b_grouping)
    axs[1].axvline(best_epoch, c='black', label='Best')
    axs[1].set_title('Mean estimate of benchmarks\' grouping')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('HDBSCAN distance')
    axs[1].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to / 'benchmarks_metrics_{}.{}'.format(parameters['id'], parameters['plots_extension']))
    pyplot.close()


def plot_variation_coefs(vc_dict, vc_dict_original, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    # save one by one for each sample in dict
    for i, type in enumerate(vc_dict):
        x = range(1, 1+len(vc_dict[type]))  # epochs
        y = vc_dict[type]  # values

        pyplot.figure()
        pyplot.plot(x, y, label='Training process')
        pyplot.hlines(y=vc_dict_original[type], xmin=x[0], xmax=x[-1], colors='r', label='Original data')
        pyplot.hlines(y=y[best_epoch-1], xmin=x[0], xmax=x[-1], colors='k', label='Normalized data')
        pyplot.vlines(x=best_epoch, ymin=min(y), ymax=y[best_epoch-1], colors='k')
        pyplot.ylabel('VC')
        pyplot.xlabel('Epochs')
        pyplot.title('Variation coefficient for {}'.format(type))
        pyplot.grid(True)
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(save_to / 'vcs_{}_{}.{}'.format(type, parameters['id'], parameters['plots_extension']))
        pyplot.close()


def plot_n_clusters(clusters_dict, clusters_dict_original, id, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    if len(clusters_dict) == 6:
        # save all on one figure
        pyplot.figure(figsize=(12, 8))
        for i, type in enumerate(clusters_dict):

            x = range(1, 1+len(clusters_dict[type]))  # epochs
            y = clusters_dict[type]  # values

            ax = pyplot.subplot(2, 3, i + 1)
            ax.plot(x,y, label='Normalized data')
            ax.axhline(y=len(set(clusters_dict_original[type])), color='r', linestyle='-', label='Original data')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Clusters found')
            ax.set_title(type)
            ax.grid(True)
            ax.legend()

        pyplot.suptitle('HDBSCAN clustering')
        pyplot.tight_layout()
        pyplot.savefig(save_to + 'clustering_{}.pdf'.format(id))
        pyplot.close()
    else:
        # save one by one for each sample in dict
        for i, type in enumerate(clusters_dict):
            x = range(1, 1+len(clusters_dict[type]))  # epochs
            y = clusters_dict[type]  # values

            pyplot.figure()
            pyplot.plot(x, y, label='Normalized data')
            pyplot.axhline(y=len(set(clusters_dict_original[type])), color='r', linestyle='-', label='Original data')
            pyplot.ylabel('Clusters found')
            pyplot.xlabel('Epochs')
            pyplot.title('HDBSCAN clustering for {}'.format(type))
            pyplot.grid(True)
            pyplot.tight_layout()
            pyplot.legend()
            pyplot.savefig(save_to + 'clustering_{}_{}.pdf'.format(type, id))
            pyplot.close()


if __name__ == '__main__':
    pass
