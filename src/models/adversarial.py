import torch, numpy, pandas, time, os, uuid, random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot
from tqdm import tqdm

from src.models.cl import Classifier
from src.models.ae import Autoencoder
from src.constants import shared_perturbations as all_samples_types
from src.constants import benchmark_sample_types as benchmarks
from src.constants import regularization_sample_types as reg_types
from src.constants import latent_dim_explained_variance_ratio as min_variance_ratio
from src.constants import loss_mapper, user, batches, min_relevant_intensity
from src.batch_analysis import compute_cv_for_samples_types, plot_batch_cross_correlations
from src.batch_analysis import compute_number_of_clusters_with_hdbscan, plot_full_dataset_umap
from src.batch_analysis import get_sample_cross_correlation_estimate


def split_to_train_and_test(values, batches, scaler, proportion=0.7):

    n_samples, n_features = values.shape

    # scale
    scaled = scaler.transform(values)

    # split values to train and val
    x_train = scaled[:int(proportion * n_samples), :]
    x_val = scaled[int(proportion * n_samples):, :]
    y_train = batches[:int(proportion * n_samples)]
    y_val = batches[int(proportion * n_samples):]

    if numpy.min(batches) == 1:
        # enumerate batches from 0 to n
        y_train -= 1
        y_val -= 1

    return x_train, x_val, y_train, y_val


def get_data(path, n_batches=None, m_fraction=None, na_fraction=None):
    # collect merged dataset
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and remove metainfo
    data = data.iloc[:, 3:].T
    data = data.fillna(min_relevant_intensity)

    if m_fraction is not None:
        # randomly select a fraction of metabolites
        all_metabolites = list(data.columns)
        metabolites_to_drop = random.sample(all_metabolites, int(round(1 - m_fraction, 2) * len(all_metabolites)))
        data = data.drop(labels=metabolites_to_drop, axis=1)

    if na_fraction is not None:
        # randomly mask a fraction of values
        data = data.mask(numpy.random.random(data.shape) < na_fraction)
        data = data.fillna(min_relevant_intensity)

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)

    if n_batches is not None:
        # select first n batches
        data = data.loc[data['batch'] <= n_batches, :]

    return data


def plot_losses(rec_loss, d_loss, g_loss, best_epoch, parameters, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

    fig, axs = pyplot.subplots(3, figsize=(6,9))

    fig.suptitle('Adversarial training loop losses')

    axs[0].plot(range(1, 1+len(d_loss)), d_loss)
    axs[0].axvline(best_epoch+1, c='black', label='Best')
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel(parameters['d_loss'])
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(g_loss)), g_loss)
    axs[1].axvline(best_epoch + 1, c='black', label='Best')
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    if parameters['use_g_regularization']:
        axs[1].set_ylabel('Regularized {} - {}'.format(parameters['g_loss'], parameters['d_loss']))
    else:
        axs[1].set_ylabel('{} - {}'.format(parameters['g_loss'], parameters['d_loss']))
    axs[1].grid(True)

    axs[2].plot(range(1, 1 + len(rec_loss)), rec_loss)
    axs[2].axvline(best_epoch + 1, c='black', label='Best')
    axs[2].set_title('Reconstruction loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel(parameters['g_loss'])
    axs[2].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to + 'losses_{}.pdf'.format(parameters['id']))


def plot_metrics(d_accuracy, reg_correlation, reg_clustering, reg_vc, best_epoch, id, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

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


def plot_benchmarks_metrics(b_correlations, b_grouping, best_epoch, id, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

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


def plot_variation_coefs(vc_dict, vc_dict_original, best_epoch, id, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

    if len(vc_dict) == 6:
        # save all on one figure
        pyplot.figure(figsize=(12, 8))
        for i, type in enumerate(vc_dict):

            x = range(1, 1+len(vc_dict[type]))  # epochs
            y = vc_dict[type]  # values

            ax = pyplot.subplot(2, 3, i + 1)
            ax.plot(x, y, label='Training process')
            ax.hlines(y=vc_dict_original[type], xmin=x[0], xmax=x[-1], colors='r', label='Original data')
            ax.hlines(y=y[best_epoch], xmin=x[0], xmax=x[-1], colors='k', label='Normalized data')
            ax.vlines(x=best_epoch+1, ymin=min(y), ymax=y[best_epoch], colors='k')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('VC')
            ax.set_title(type)
            ax.grid(True)
            ax.legend()

        pyplot.suptitle('Variation coefficients')
        pyplot.tight_layout()
        pyplot.savefig(save_to + 'vcs_{}.pdf'.format(id))
    else:
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
            pyplot.tight_layout()
            pyplot.savefig(save_to + 'vcs_{}_{}.pdf'.format(type, id))


def plot_n_clusters(clusters_dict, clusters_dict_original, id, save_to='/Users/{}/ETH/projects/normalization/res/'.format(user)):

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


def slice_by_grouping_and_correlation(history, g_percent, c_percent):
    try:
        # grouping slice
        df = history[history['reg_grouping'] <= numpy.percentile(history['reg_grouping'].values, g_percent)].sort_values('reg_grouping')
        # correlation slice + sorting by variation coefs
        df = df[df['reg_corr'] >= numpy.percentile(df['reg_corr'].values, c_percent)].sort_values('reg_vc')
        # negative loss slice (desired by model construction)
        df = df[df['g_loss'] < 0]
        assert df.shape[0] > 0

    except Exception:
        df = None
    return df


def find_best_epoch(history, skip_epochs=10):
    # skip first n epochs
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
                        min_grouping_epoch = int(history.loc[history['reg_grouping'] == history['reg_grouping'].min(), 'epoch'].values[-1])
                        print('WARNING: couldn\'t find the best epoch, returning the last one of min grouping coef: epoch {}'.format(min_grouping_epoch+1))
                        return min_grouping_epoch
    return int(df['epoch'].values[0])


def define_latent_dim_with_pca(data):

    transformer = PCA()
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    transformer.fit(scaled_data)

    for i in range(0, len(transformer.explained_variance_ratio_), 5):
        if sum(transformer.explained_variance_ratio_[:i]) > min_variance_ratio:
            return i


def run_normalization(data, parameters):

    if int(parameters['latent_dim']) < 0:
        # latent_dim is defined by PCA and desired level of variance explained
        parameters['latent_dim'] = define_latent_dim_with_pca(data)

    # create models
    device = torch.device("cpu")
    discriminator = Classifier(input_shape=int(parameters['latent_dim']), n_batches=int(parameters['n_batches'])).to(device)
    generator = Autoencoder(input_shape=int(parameters['n_features']), latent_dim=int(parameters['latent_dim'])).to(device)

    print('Discriminator:\n', discriminator)
    print('Number of parameters: ', discriminator.count_parameters())
    print('Generator:\n', generator)
    print('Number of parameters: ', generator.count_parameters())

    # create an optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=float(parameters['d_lr']))
    g_optimizer = optim.Adam(generator.parameters(), lr=float(parameters['g_lr']))

    # define losses
    d_criterion = loss_mapper[parameters['d_loss']]
    g_criterion = loss_mapper[parameters['g_loss']]

    # split to values and batches
    data_batch_labels = data.iloc[:, 0]
    data_values = data.iloc[:, 1:]

    # get CV of benchmarks in original data
    cv_dict_original = compute_cv_for_samples_types(data_values, sample_types_of_interest=benchmarks)

    # create and fit the scaler
    scaler = RobustScaler().fit(data_values)
    # apply scaling and do train test split
    X_train, X_test, y_train, y_test = split_to_train_and_test(data_values, data_batch_labels, scaler, proportion=float(parameters['train_ratio']))

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=int(parameters['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(parameters['batch_size']), shuffle=False)

    # create folders to save results
    save_to = parameters['out_path'] + '{}/'.format(parameters['id'])
    if not os.path.exists(save_to):
        os.makedirs(save_to)
        os.makedirs(save_to + '/callbacks')
        os.makedirs(save_to + '/checkpoints')
        os.makedirs(save_to + '/benchmarks')

    # Lists to keep track of progress
    g_loss_history = []
    d_loss_history = []
    rec_loss_history = []

    val_acc_history = []
    reg_samples_grouping_history = []
    reg_samples_corr_history = []
    reg_samples_vc_history = []

    benchmarks_corr_history = []
    benchmarks_grouping_history = []
    benchmarks_variation_coefs = dict([(sample, []) for sample in benchmarks])

    g_regularizer = 0
    total_epochs = int(parameters['g_epochs']) + int(parameters['d_epochs']) + int(parameters['adversarial_epochs'])
    for epoch in range(total_epochs+1):

        start = time.time()
        d_loss_per_epoch = 0
        g_loss_per_epoch = 0
        rec_loss_per_epoch = 0

        for batch_features, labels in train_loader:

            # TRAIN DISCRIMINATOR
            batch_features = batch_features.to(device)
            # generate encodings
            encodings = generator.encode(batch_features)

            d_optimizer.zero_grad()
            # compute reconstructions
            predictions = discriminator(encodings)

            # compute training reconstruction loss
            d_loss = d_criterion(predictions, labels)
            # compute accumulated gradients
            d_loss.backward()
            d_loss_per_epoch += d_loss.item()
            # perform parameter update based on current gradients
            d_optimizer.step()

            # TRAIN GENERATOR
            g_optimizer.zero_grad()
            with torch.enable_grad():
                g_loss = 0.
                # compute reconstructions
                reconstruction = generator(batch_features)
                # compute training reconstruction loss
                reconstruction_loss = g_criterion(reconstruction, batch_features)
                rec_loss_per_epoch += reconstruction_loss.item()

                # add regularization by grouping of benchmarks
                g_loss += (1 + g_regularizer) * reconstruction_loss
                # substitute discriminator loss to push it towards smaller batch effects
                g_loss -= float(parameters['d_lambda']) * d_loss.item()

                # compute accumulated gradients
                g_loss.backward()
                g_loss_per_epoch += g_loss.item()

                # perform parameter update based on current gradients
                g_optimizer.step()

        # COMPUTE EPOCH LOSSES
        d_loss = d_loss_per_epoch / len(train_loader)
        g_loss = g_loss_per_epoch / len(train_loader)
        rec_loss = rec_loss_per_epoch / len(train_loader)

        d_loss_history.append(d_loss)
        g_loss_history.append(g_loss)
        rec_loss_history.append(rec_loss)

        # GENERATE DATA FOR OTHER METRICS
        scaled_data_values = scaler.transform(data_values.values)

        encodings = generator.encode(torch.Tensor(scaled_data_values))
        reconstruction = generator.decode(encodings)

        encodings = pandas.DataFrame(encodings.detach().numpy(), index=data_values.index)
        encodings.insert(0, 'batch', data_batch_labels)

        reconstruction = scaler.inverse_transform(reconstruction.detach().numpy())
        reconstruction = pandas.DataFrame(reconstruction, index=data_values.index)

        # COMPUTE METRICS
        # classification accuracy of TEST data
        accuracy = []
        for batch_features, batch_labels in test_loader:

            batch_features = batch_features.to(device)
            # generate encodings
            batch_encodings = generator.encode(batch_features)
            batch_predictions = discriminator(batch_encodings)

            # calculate accuracy per batch
            true_positives = (batch_predictions.argmax(-1) == batch_labels).float().detach().numpy()
            batch_accuracy = true_positives.sum() / len(true_positives)
            accuracy.append(batch_accuracy)

        accuracy = numpy.mean(accuracy)  # save for printing
        val_acc_history.append(accuracy)  # save for plotting

        # collect variation coefficients for some samples of ALL reconstructed data
        vcs = compute_cv_for_samples_types(reconstruction, sample_types_of_interest=all_samples_types)
        reg_vcs_sum = 0.
        for sample in all_samples_types:
            if sample in reg_types:
                reg_vcs_sum += vcs[sample]  # calculate sum of (reg) variation coefs
            if sample in benchmarks:
                benchmarks_variation_coefs[sample].append(vcs[sample])  # append vcs of benchmarks

        reg_vc = reg_vcs_sum / len(vcs)  # compute mean overall variation coef
        reg_samples_vc_history.append(reg_vc)

        # collect clustering results for some samples of ALL encoded data
        clustering, total_clusters = compute_number_of_clusters_with_hdbscan(encodings, parameters, print_info=False, sample_types_of_interest=reg_types)

        # assess cross correlations of regularization samples in ALL reconstructed data
        reg_corr = get_sample_cross_correlation_estimate(reconstruction, percent=25, sample_types_of_interest=reg_types)
        reg_samples_corr_history.append(reg_corr)

        # assess cross correlations of benchmarks in ALL reconstructed data
        b_corr = get_sample_cross_correlation_estimate(reconstruction, sample_types_of_interest=benchmarks)
        benchmarks_corr_history.append(b_corr)

        # assess grouping of samples: compute g_lambda, so that it equals
        # 0, when all samples of a reg_type belong to the sample cluster
        # 1, when N samples of a reg_type belong to N different clusters
        b_grouping_coefs = []
        reg_grouping_coefs = []
        for sample in reg_types:

            n_sample_clusters = len(set(clustering[sample]))
            max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
            coef = (n_sample_clusters - 1) / max_n_clusters  # minus 1 to account for uncertainty in HDBSCAN
            reg_grouping_coefs.append(coef)
            if sample in benchmarks:
                b_grouping_coefs.append(coef)  # append a coef for a benchmark

        b_grouping = numpy.mean(b_grouping_coefs)
        reg_grouping = numpy.mean(reg_grouping_coefs)  # there are more samples here, I assume

        benchmarks_grouping_history.append(b_grouping)
        reg_samples_grouping_history.append(reg_grouping)

        # SET REGULARIZATION FOR GENERATOR'S NEXT ITERATION
        if parameters['use_g_regularization']:
            g_regularizer = float(parameters['g_lambda']) * reg_grouping

        # SAVE MODEL
        torch.save(generator.state_dict(), save_to + '/checkpoints/ae_at_{}_{}.models'.format(epoch, parameters['id']))

        # PRINT AND PLOT EPOCH INFO
        # plot every N epochs what happens with data
        if int(parameters['callback_step']) > 0 and epoch % int(parameters['callback_step']) == 0:
            # plot cross correlations of benchmarks in ALL reconstructed data
            plot_batch_cross_correlations(reconstruction, 'epoch {}'.format(epoch+1), parameters['id'], sample_types_of_interest=benchmarks, save_to=save_to+'/callbacks/', save_plot=True)
            # plot umap of FULL encoded data
            plot_full_dataset_umap(encodings, 'epoch {}'.format(epoch+1), parameters, save_to=save_to+'/callbacks/')
            pyplot.close('all')

        # display the epoch training loss
        timing = int(time.time() - start)
        print("epoch {}/{}, {} sec elapsed:\n"
              "g_loss = {:.4f}, rec_loss = {:.4f}, d_loss = {:.4f}, "
              "val_acc = {:.4f}, reg_grouping = {:.4f}, reg_corr = {:.4f}, reg_vc = {:.4f}\n".format(epoch + 1, total_epochs, timing, g_loss, rec_loss, d_loss, accuracy, reg_grouping, reg_corr, reg_vc))

    # PLOT TRAINING HISTORY
    history = pandas.DataFrame({'epoch': [x for x in range(len(d_loss_history))], 'best': [False for x in range(len(d_loss_history))],
                                'rec_loss': rec_loss_history, 'd_loss': d_loss_history, 'g_loss': g_loss_history,
                                'reg_grouping': reg_samples_grouping_history, 'reg_corr': reg_samples_corr_history, 'reg_vc': reg_samples_vc_history,
                                'val_acc': val_acc_history, 'b_corr': benchmarks_corr_history, 'b_grouping': benchmarks_grouping_history})

    best_epoch = find_best_epoch(history, skip_epochs=parameters['skip_epochs'])
    history.loc[best_epoch, 'best'] = True  # mark the best epoch

    plot_losses(rec_loss_history, d_loss_history, g_loss_history, best_epoch, parameters, save_to=save_to)
    plot_metrics(val_acc_history, reg_samples_corr_history, reg_samples_grouping_history, reg_samples_vc_history,
                 best_epoch, parameters['id'], save_to=save_to)

    plot_benchmarks_metrics(benchmarks_corr_history, benchmarks_grouping_history, best_epoch, parameters['id'], save_to=save_to+'/benchmarks/')
    plot_variation_coefs(benchmarks_variation_coefs, cv_dict_original, best_epoch, parameters['id'], save_to=save_to+'/benchmarks/')

    # LOAD BEST MODEL
    generator = Autoencoder(input_shape=int(parameters['n_features']), latent_dim=int(parameters['latent_dim'])).to(device)
    generator.load_state_dict(torch.load(save_to + 'checkpoints/ae_at_{}_{}.models'.format(best_epoch, parameters['id']), map_location=device))
    generator.eval()

    # PLOT BEST EPOCH CALLBACKS
    scaled_data_values = scaler.transform(data_values.values)

    encodings = generator.encode(torch.Tensor(scaled_data_values))
    reconstruction = generator.decode(encodings)

    encodings = pandas.DataFrame(encodings.detach().numpy(), index=data_values.index)
    encodings.insert(0, 'batch', data_batch_labels)

    reconstruction = scaler.inverse_transform(reconstruction.detach().numpy())
    reconstruction = pandas.DataFrame(reconstruction, index=data_values.index)

    # plot cross correlations of benchmarks in ALL reconstructed data
    plot_batch_cross_correlations(reconstruction, 'best model at {}'.format(best_epoch + 1), parameters['id'], sample_types_of_interest=benchmarks, save_to=save_to+'/benchmarks/', save_plot=True)
    # plot umap of FULL encoded data
    plot_full_dataset_umap(encodings, 'best model at {}'.format(best_epoch + 1), parameters, save_to=save_to)
    pyplot.close('all')

    # SAVE ENCODED AND NORMALIZED DATA
    encodings.to_csv(save_to + 'encodings_{}.csv'.format(parameters['id']))
    reconstruction.to_csv(save_to + 'normalized_{}.csv'.format(parameters['id']))

    # SAVE PARAMETERS AND HISTORY
    pandas.DataFrame(parameters, index=[0]).to_csv(save_to + 'parameters_{}.csv'.format(parameters['id']), index=False)
    history.to_csv(save_to + 'history_{}.csv'.format(parameters['id']), index=False)

    # REFACTOR CHECKPOINTS
    for file in os.listdir(save_to + '/checkpoints/'):
        if file.startswith('ae_at_{}_'.format(best_epoch)):
            # rename to best
            os.rename(save_to + '/checkpoints/' + file, save_to + '/checkpoints/best_' + file)
        else:
            if not parameters['keep_checkpoints']:
                # remove the rest
                os.remove(save_to + '/checkpoints/' + file)


if __name__ == "__main__":

    for i in tqdm(range(50)):

        # PARAMETERS
        parameters = {

            'in_path': '/Users/{}/ETH/projects/normalization/data/'.format(user),
            'out_path': '/Users/{}/ETH/projects/normalization/res/'.format(user),
            'id': str(uuid.uuid4())[:8],

            'n_features': 170,  # n of metabolites in initial dataset
            'latent_dim': -1,  # n dimensions to reduce to (50 makes 99% of variance in PCA)
            'n_batches': 7,
            'n_replicates': 3,

            'd_lr': 0.0014,  # discriminator learning rate
            'g_lr': 0.0001,  # generator learning rate
            'd_loss': 'CE',
            'g_loss': 'MSE',
            'd_lambda': 8,  # discriminator regularization term coefficient
            'g_lambda': 2.4,  # generator regularization term coefficient
            'use_g_regularization': True,  # whether to use generator regularization term
            'train_ratio': 0.9,  # for train-test split
            'batch_size': 64,
            'g_epochs': 0,  # pretraining of generator (not implemented)
            'd_epochs': 0,  # pretraining of discriminator (not implemented)
            'adversarial_epochs': 50,  # simultaneous competitive training

            'skip_epochs': 5,  # number of epochs to skip before looking for the best
            'callback_step': -1,  # save callbacks every n epochs
            'keep_checkpoints': False  # whether to keep all checkpoints, or just the best epoch
        }

        data = get_data(parameters['in_path'], na_fraction=0.1)
        run_normalization(data, parameters)
