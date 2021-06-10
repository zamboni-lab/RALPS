
import seaborn, pandas, numpy, os
from matplotlib import pyplot

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

    best_models = pandas.DataFrame()
    for id in os.listdir(config['out_path']):
        if id.startswith('.'):
            pass
        else:
            id_results = pandas.read_csv(config['out_path'] + id + '/history_{}.csv'.format(id))
            id_results = id_results.loc[id_results['best'] == True, :]
            id_results['id'] = id
            best_models = pandas.concat([best_models, id_results])
            del id_results
    best_models['best'] = False

    # pick best models
    print('GRID SEARCH BEST MODELS:', '\n')
    top = slice_by_grouping_and_correlation(best_models, g_percent, c_percent)
    if top is None:
        print('WARNING: could not find the best model, returning the list sorted by reg_grouping\n')
        best_models = best_models.sort_values('reg_grouping')
        print(best_models.to_string(), '\n')

    else:
        for top_id in top['id'].values:
            # mark the best models
            best_models.loc[best_models['id'] == top_id, 'best'] = True
        print(top.to_string(), '\n')

    best_models.to_csv(config['out_path'] + 'best_models.csv', index=False)
    print('full grid saved to {}best_models.csv'.format(config['out_path']))


def slice_by_grouping_and_correlation(history, g_percent, c_percent):
    """ This method is used to select the best epoch or the best model.
        It applies two thresholds and sorting for the three key quality metrics. """

    try:
        # grouping slice
        df = history[history['reg_grouping'] <= numpy.percentile(history['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        # negative loss slice (desired by model construction)
        df = df[df['g_loss'] < 0]
        df['best'] = True
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def find_best_epoch(history, skip_epochs=5):
    """ This method seeks for the best epoch using logged history of quality metrics. """

    # skip first n epochs
    if 0 < skip_epochs < history.shape[0]:
        history = history.iloc[skip_epochs:, :]

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
                            print('WARNING: couldn\'t find the best epoch,'
                                  'returning the one of min grouping with max cross-correlation: epoch {}'.format(best_epoch + 1))
                        else:
                            # min grouping
                            best_epoch = int(history.loc[history['reg_grouping'] == history['reg_grouping'].min(), 'epoch'].values[-1])
                            print('WARNING: couldn\'t find the best epoch,'
                                  'returning the last one of min grouping coef: epoch {}'.format(best_epoch + 1))

                        return best_epoch
    return int(df['epoch'].values[0])


def plot_losses(rec_loss, d_loss, g_loss, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(3, figsize=(6,9))

    fig.suptitle('Adversarial training loop losses')

    axs[0].plot(range(1, 1+len(d_loss)), d_loss)
    axs[0].axvline(best_epoch+1, c='black', label='Best')
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Cross-entropy')
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(g_loss)), g_loss)
    axs[1].axvline(best_epoch + 1, c='black', label='Best')
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Regularized MSE - Cross-entropy')
    axs[1].grid(True)

    axs[2].plot(range(1, 1 + len(rec_loss)), rec_loss)
    axs[2].axvline(best_epoch + 1, c='black', label='Best')
    axs[2].set_title('Reconstruction loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MSE')
    axs[2].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to + 'losses_{}.pdf'.format(parameters['id']))


def plot_metrics(d_accuracy, reg_correlation, reg_clustering, reg_vc, best_epoch, id, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(2, 2, figsize=(9,6))

    fig.suptitle('Adversarial training loop metrics')

    axs[0,0].plot(range(1, 1+len(d_accuracy)), d_accuracy)
    axs[0,0].axvline(best_epoch + 1, c='black', label='Best')
    axs[0,0].set_title('Validation classifier accuracy')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Accuracy')
    axs[0,0].grid(True)

    axs[1,0].plot(range(1, 1 + len(reg_correlation)), reg_correlation)
    axs[1,0].axvline(best_epoch + 1, c='black', label='Best')
    axs[1,0].set_title('Sum of samples\' cross-correlation estimates')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Pearson coef')
    axs[1,0].grid(True)

    axs[0,1].plot(range(1, 1 + len(reg_clustering)), reg_clustering)
    axs[0,1].axvline(best_epoch + 1, c='black', label='Best')
    axs[0,1].set_title('Mean estimate of samples\' grouping')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('HBDSCAN distance')
    axs[0,1].grid(True)

    axs[1,1].plot(range(1, 1+len(reg_vc)), reg_vc)
    axs[1,1].axvline(best_epoch + 1, c='black', label='Best')
    axs[1,1].set_title('Mean samples\' variation coef')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('VC')
    axs[1,1].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to + 'metrics_{}.pdf'.format(id))


def plot_benchmarks_metrics(b_correlations, b_grouping, best_epoch, id, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(2, figsize=(6,6))

    fig.suptitle('Benchmarks metrics')

    axs[0].plot(range(1, 1+len(b_correlations)), b_correlations)
    axs[0].axvline(best_epoch+1, c='black', label='Best')
    axs[0].set_title('Sum of benchmarks cross-correlation')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Pearson coef')
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(b_grouping)), b_grouping)
    axs[1].axvline(best_epoch + 1, c='black', label='Best')
    axs[1].set_title('Mean estimate of benchmarks\' grouping')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('HDBSCAN distance')
    axs[1].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to + 'benchmarks_metrics_{}.pdf'.format(id))


def plot_variation_coefs(vc_dict, vc_dict_original, best_epoch, id, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    # save one by one for each sample in dict
    for i, type in enumerate(vc_dict):
        x = range(1, 1+len(vc_dict[type]))  # epochs
        y = vc_dict[type]  # values

        pyplot.figure()
        pyplot.plot(x, y, label='Training process')
        pyplot.hlines(y=vc_dict_original[type], xmin=x[0], xmax=x[-1], colors='r', label='Original data')
        pyplot.hlines(y=y[best_epoch], xmin=x[0], xmax=x[-1], colors='k', label='Normalized data')
        pyplot.vlines(x=best_epoch+1, ymin=min(y), ymax=y[best_epoch], colors='k')
        pyplot.ylabel('VC')
        pyplot.xlabel('Epochs')
        pyplot.title('Variation coefficient for {}'.format(type))
        pyplot.grid(True)
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(save_to + 'vcs_{}_{}.pdf'.format(type, id))


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
