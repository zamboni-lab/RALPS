
import seaborn, pandas, numpy, os, torch
from matplotlib import pyplot
from pathlib import Path

from models.ae import Autoencoder
from constants import grouping_threshold_percent as g_percent
from constants import correlation_threshold_percent as c_percent
from constants import n_best_models


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

    if best_models.shape[0] == 0:
        print('WARNING: no solutions found! Check input data, try other parameters, or report an issue.\n')
    else:

        ids = best_models['id']
        best_models = best_models.drop(columns=['solution', 'id', 'g_loss', 'val_acc', 'stopped_early'])
        best_models.insert(1, 'best', False)
        best_models.insert(2, 'id', ids)

        print('GRID SEARCH BEST MODELS:\n')
        top = select_top_solutions(best_models, g_percent, c_percent)
        if top is None:
                print('WARNING: could not find the best solution, returning the full list sorted by rec_loss\n')
                best_models = best_models.sort_values('rec_loss')
                print(best_models.to_string(), '\n')
        else:
            for top_id in top['id'].values:
                # mark the best models
                best_models.loc[best_models['id'] == top_id, 'best'] = True
            print(top.to_string(), '\n')

        best_models = best_models.sort_values(['best', 'rec_loss'], ascending=[False, True])
        best_models = best_models.round({key: 3 for key in best_models.columns[3:] if key != 'ivc_percent'})
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
        df = pandas.concat([df, history[
            (history['all_grouping'] <= numpy.percentile(history['all_grouping'].values, g_percent))
            & (history['ivc_percent'] <= numpy.percentile(history['ivc_percent'].values, g_percent))
            ]])

        df = df.drop_duplicates().sort_values('rec_loss')
        df = df.iloc[:n_best_models, :]  # keep only n best

        assert df.shape[0] > 0
        df['best'] = True
    except Exception:
        df = None
    return df


def find_best_epoch(history, skip_epochs, mean_batch_vc_initial, mean_reg_vc_initial):
    """ This method seeks for the best epoch using logged history of quality metrics. """

    # skip first n epochs
    if 0 < skip_epochs:
        if skip_epochs < history.shape[0]:
            history = history.iloc[skip_epochs:, :]
        else:
            print('WARNING: {} epochs skipped of total {} -> no solution for current parameter set\n'
                  .format(skip_epochs, history.shape[0]))
            return -1

    # filter out epochs of high reconstruction error
    history = history.loc[history['rec_loss'] < history['rec_loss'].values[0] / 2, :]
    if history.shape[0] < 1:
        print('WARNING: low reconstruction quality -> no solution for current parameter set\n')
        return -1

    # filter out epochs of increased variation coefs
    history = history.loc[(history['batch_vc'] < mean_batch_vc_initial) & (history['reg_vc'] < mean_reg_vc_initial), :]
    if history.shape[0] < 1:
        print('WARNING: increased VCs -> no solution for current parameter set\n')
        return -1

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
                                      'returning the one of min grouping with max cross-correlation: epoch {}\n'.format(best_epoch))
                            else:
                                # min grouping
                                best_epoch = int(history.loc[history['reg_grouping'] == history['reg_grouping'].min(), 'epoch'].values[-1])
                                print('WARNING: couldn\'t find the best epoch, '
                                      'returning the last one of min grouping coef: epoch {}\n'.format(best_epoch))
                            return best_epoch

        return int(df['epoch'].values[0])


def plot_losses(rec_loss, d_loss, g_loss, v_loss, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(2, 2, figsize=(9,6))

    fig.suptitle('Adversarial training loop losses')

    axs[0,0].plot(range(1, 1+len(d_loss)), d_loss)
    axs[0,0].axvline(best_epoch, c='black', label='Best')
    axs[0,0].set_title('Classifier loss')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Cross-entropy')
    axs[0,0].grid(True)

    axs[1,0].plot(range(1, 1+len(g_loss)), g_loss)
    axs[1,0].axvline(best_epoch, c='black', label='Best')
    axs[1,0].set_title('Autoencoder loss')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Regularized MSE - Cross-entropy')
    axs[1,0].grid(True)

    axs[0,1].plot(range(1, 1 + len(rec_loss)), rec_loss)
    axs[0,1].axvline(best_epoch, c='black', label='Best')
    axs[0,1].set_title('Reconstruction loss')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('MSE')
    axs[0,1].grid(True)

    axs[1,1].plot(range(1, 1 + len(v_loss)), v_loss)
    axs[1,1].axvline(best_epoch, c='black', label='Best')
    axs[1,1].set_title('Variation loss')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('Adjusted median diff')
    axs[1,1].grid(True)

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


def plot_variation_coefs(vc_dict, vc_dict_initial, best_epoch, parameters, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    # save one by one for each sample in dict
    for i, type in enumerate(vc_dict):
        x = range(1, 1+len(vc_dict[type]))  # epochs
        y = vc_dict[type]  # values

        pyplot.figure()
        pyplot.plot(x, y, label='Training')
        pyplot.hlines(y=vc_dict_initial[type], xmin=x[0], xmax=x[-1], colors='r', label='Initial data')
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


def plot_n_clusters(clusters_dict, clusters_dict_initial, id, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    if len(clusters_dict) == 6:
        # save all on one figure
        pyplot.figure(figsize=(12, 8))
        for i, type in enumerate(clusters_dict):

            x = range(1, 1+len(clusters_dict[type]))  # epochs
            y = clusters_dict[type]  # values

            ax = pyplot.subplot(2, 3, i + 1)
            ax.plot(x,y, label='Normalized data')
            ax.axhline(y=len(set(clusters_dict_initial[type])), color='r', linestyle='-', label='Initial data')
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
            pyplot.axhline(y=len(set(clusters_dict_initial[type])), color='r', linestyle='-', label='Initial data')
            pyplot.ylabel('Clusters found')
            pyplot.xlabel('Epochs')
            pyplot.title('HDBSCAN clustering for {}'.format(type))
            pyplot.grid(True)
            pyplot.tight_layout()
            pyplot.legend()
            pyplot.savefig(save_to + 'clustering_{}_{}.pdf'.format(type, id))
            pyplot.close()


def evaluate_checkpoints(path_to_weights, device='cpu'):
    """ This method loads the weights of the selected checkpoints
        and evaluates their normalization effects on the data.
        NB: the weights directory is assumed to be consistent with the default RALPS output. """

    path = Path(path_to_weights)

    if os.path.exists(path):
        checkpoints = [x for x in os.listdir(path) if x.endswith('.torch')]
        if len(checkpoints) > 0:
            # read the corresponding parameters
            parameters = pandas.read_csv(path.parent / 'parameters_{}.csv'.format(path.parent.name), index_col=0).to_dict()['values']
            parameters['n_features'] = int(parameters['n_features'])
            parameters['latent_dim'] = int(parameters['latent_dim'])
            parameters['n_replicates'] = int(parameters['n_replicates'])
            parameters['n_batches'] = int(parameters['n_batches'])
            parameters['min_relevant_intensity'] = int(parameters['min_relevant_intensity'])
            # parse out the samples used for benchmarking
            benchmarks = parameters['benchmarks'].split(',') if parameters['benchmarks'] != '' else []

            from ralps import get_data
            data = get_data(parameters, parameters)
            # split to values and batches
            data_batch_labels = data.iloc[:, 0]
            data_values = data.iloc[:, 1:]

            # create and fit the scaler
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(data_values)
            scaled_data_values = scaler.transform(data_values.values)

            for cp in checkpoints:
                # create a folder for a checkpoint
                save_to = path / cp.replace('.torch', '')
                if not os.path.exists(save_to):
                    os.makedirs(save_to)

                # LOAD MODEL
                generator = Autoencoder(input_shape=parameters['n_features'], latent_dim=parameters['latent_dim']).to(device)
                generator.load_state_dict(torch.load(path / cp, map_location=device))
                generator.eval()

                # APPLY NORMALIZATION AND PLOT RESULTS
                encodings = generator.encode(torch.Tensor(scaled_data_values).to(device))
                reconstruction = generator.decode(encodings)

                encodings = pandas.DataFrame(encodings.detach().cpu().numpy(), index=data_values.index)
                reconstruction = scaler.inverse_transform(reconstruction.detach().cpu().numpy())
                reconstruction = pandas.DataFrame(reconstruction, index=data_values.index, columns=data_values.columns)
                from src import evaluation
                reconstruction = evaluation.mask_non_relevant_intensities(reconstruction, parameters['min_relevant_intensity'])

                from src import batch_analysis
                # plot umaps of initial and normalized data
                batch_analysis.plot_full_data_umaps(data_values, reconstruction, data_batch_labels, parameters, save_to=save_to)
                # plot batch variation coefs in initial and normalized data
                vc_batch_original = batch_analysis.compute_vc_for_batches(data_values, data_batch_labels)
                vc_batch_normalized = batch_analysis.compute_vc_for_batches(reconstruction, data_batch_labels)
                batch_analysis.plot_batch_vcs(vc_batch_original, vc_batch_normalized, parameters, save_to=save_to)

                if len(benchmarks) > 0:
                    if not os.path.exists(save_to / 'benchmarks'):
                        os.makedirs(save_to / 'benchmarks')
                    # plot cross correlations of benchmarks in initial and normalized data
                    batch_analysis.plot_batch_cross_correlations(data_values, 'initial', parameters, benchmarks, save_to=save_to / 'benchmarks', save_plot=True)
                    batch_analysis.plot_batch_cross_correlations(reconstruction, 'normalized', parameters, benchmarks, save_to=save_to / 'benchmarks', save_plot=True)

                # SAVE ENCODED AND NORMALIZED DATA
                encodings.to_csv(save_to / 'encodings_{}.csv'.format(parameters['id']))
                from src import processing
                reconstruction.index = processing.get_initial_samples_names(reconstruction.index)  # reindex to original names
                reconstruction.T.to_csv(save_to / 'normalized_{}.csv'.format(parameters['id']))
                print('results saved to {}'.format(save_to))
        else:
            print('No checkpoint is found to evaluate.')
    else:
        print('No checkpoint is found to evaluate.')


def check_paths_for_filtering(path_to_data):
    """ This method checks that all necessary paths exist. """

    parameters_path = None
    best_model_path = None
    flag = True  # assume all files are there

    if not os.path.exists(path_to_data):
        print("Missing files to perform filtering:")
        print('- Data file is not found.')
        flag = False

    if os.path.exists(path_to_data.parent / 'parameters_{}.csv'.format(path_to_data.parent.name)):
        # main results directory
        parameters_path = path_to_data.parent / 'parameters_{}.csv'.format(path_to_data.parent.name)

        if os.path.exists(path_to_data.parent / 'checkpoints'):
            cp_paths = [path_to_data.parent / 'checkpoints' / x for x in os.listdir(path_to_data.parent / 'checkpoints') if
                        x.startswith('best') and x.endswith('.torch')]
            if len(cp_paths) > 0:
                best_model_path = cp_paths[0]
            else:
                if flag:
                    print("Missing files to perform filtering:")
                print('- Checkpoint file (best model) is not found.')
                flag = False
        else:
            if flag:
                print("Missing files to perform filtering:")
            print('- Checkpoint file (best model) is not found.')
            flag = False

    elif path_to_data.parent.parent.name == 'checkpoints':
        # directory of a particular checkpoint

        if os.path.exists(path_to_data.parent.parent.parent / 'parameters_{}.csv'.format(path_to_data.parent.parent.parent.name)):
            parameters_path = path_to_data.parent.parent.parent / 'parameters_{}.csv'.format(path_to_data.parent.parent.parent.name)
        else:
            if flag:
                print("Missing files to perform filtering:")
            print('- Parameters file is not found.')
            flag = False

        if os.path.exists(Path(str(path_to_data.parent) + '.torch')):
            best_model_path = Path(str(path_to_data.parent) + '.torch')
        else:
            if flag:
                print("Missing files to perform filtering:")
            print('- Checkpoint file (best model) is not found.')
            flag = False
    else:
        if flag:
            print("Missing files to perform filtering:")
        print('- Parameters file is not found.')
        flag = False

    return parameters_path, best_model_path, flag


def remove_outliers(path_to_data, device='cpu'):
    """ This method removes outliers from the normalized data using a variant of a boxplot outlier detection. """

    parameters_path, best_model_path, all_good = check_paths_for_filtering(Path(path_to_data))

    if all_good:
        # read the corresponding parameters
        parameters = pandas.read_csv(parameters_path, index_col=0).to_dict()['values']
        parameters['n_features'] = int(parameters['n_features'])
        parameters['latent_dim'] = int(parameters['latent_dim'])
        parameters['n_replicates'] = int(parameters['n_replicates'])
        parameters['n_batches'] = int(parameters['n_batches'])
        parameters['min_relevant_intensity'] = int(parameters['min_relevant_intensity'])
        parameters['allowed_vc_increase'] = float(parameters['allowed_vc_increase'])

        from ralps import get_data
        data = get_data(parameters, parameters)
        data_values = data.iloc[:, 1:]

        # create and fit the scaler
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(data_values)
        scaled_data_values = scaler.transform(data_values.values)

        # LOAD MODEL
        generator = Autoencoder(input_shape=parameters['n_features'], latent_dim=parameters['latent_dim']).to(device)
        generator.load_state_dict(torch.load(best_model_path, map_location=device))
        generator.eval()

        # APPLY NORMALIZATION
        encodings = generator.encode(torch.Tensor(scaled_data_values).to(device))
        reconstruction = generator.decode(encodings)

        reconstruction = scaler.inverse_transform(reconstruction.detach().cpu().numpy())
        reconstruction = pandas.DataFrame(reconstruction, index=data_values.index, columns=data_values.columns)
        from src import evaluation
        reconstruction = evaluation.mask_non_relevant_intensities(reconstruction, parameters['min_relevant_intensity'])

        factor = 1.5  # default boxplot method
        sample_percent = 100  # as if all samples had increased VCs
        while int(sample_percent) > 0:
            filtered, sample_percent, metabolite_mean = filter_outliers_with_boxplot_iqr_factor(data_values, reconstruction, iqr_factor=factor,
                                                                                                allowed_percent=parameters['allowed_vc_increase'])
            factor = int(factor + 1)

        from src import processing
        reconstruction.index = processing.get_initial_samples_names(reconstruction.index)  # reindex to original names
        reconstruction.T.to_csv(Path(path_to_data).parent / 'filtered_normalized_{}.csv'.format(parameters['id']))
        print('Filtering is complete with iqr_factor={}.'.format(factor))
        print('{} metabolites per sample were dropped (on average).'.format(int(metabolite_mean)))


def filter_outliers_with_boxplot_iqr_factor(initial, normalized, iqr_factor=1.5, allowed_percent=0.05):
    """ This method applies a variant of boxplot outlier removal for each sample in the normalized data,
        if it has an increased VC compared to the initial data. Outliers are  replaced with numpy.nan. """

    count = 0  # of samples with increased VCs found
    metabolites_lost = []  # of metabolites dropped after filtering
    for i in range(initial.shape[0]):

        init_vc = initial.iloc[i, :].std() / initial.iloc[i, :].mean()
        sample_vc = normalized.iloc[i, :].std() / normalized.iloc[i, :].mean()

        if sample_vc - init_vc > init_vc * allowed_percent:

            # filter with boxplot
            sample = normalized.iloc[i, :]
            q1 = numpy.percentile(sample, 25)
            q3 = numpy.percentile(sample, 75)
            iqr = q3 - q1
            filtered_sample = sample.copy()

            # mark outliers with nans
            filtered_sample[~((sample > q1 - iqr_factor * iqr) & (sample < q3 + iqr_factor * iqr))] = numpy.nan

            # recalculate sample vc after filtering
            sample_vc = filtered_sample.std() / filtered_sample.mean()
            metabolites_lost.append(filtered_sample.isna().sum())

            normalized.iloc[i, :] = filtered_sample

            if sample_vc - init_vc > init_vc * allowed_percent:
                count += 1

    percent_of_increased_vc = int(count / normalized.shape[0] * 100)
    mean_number_of_dropped_metabolites = numpy.mean(metabolites_lost)

    return normalized, percent_of_increased_vc, mean_number_of_dropped_metabolites


if __name__ == '__main__':
    pass
