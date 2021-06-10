import torch, numpy, pandas, time, os, uuid, random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot
from tqdm import tqdm

from models.cl import Classifier
from models.ae import Autoencoder
import evaluation, batch_analysis, preprocessing


def run_normalization(data, parameters):

    # create folders to save results
    save_to = parameters['out_path'] + '{}/'.format(parameters['id'])
    if not os.path.exists(save_to):
        os.makedirs(save_to)
        os.makedirs(save_to + '/callbacks')
        os.makedirs(save_to + '/checkpoints')
        os.makedirs(save_to + '/benchmarks')

    # parse samples of interest
    reg_types = parameters['reg_types'].split(',')
    benchmarks = parameters['benchmarks'].split(',')
    all_samples_types = [*benchmarks, *reg_types]

    # create models
    device = torch.device("cpu")
    discriminator = Classifier(input_shape=parameters['latent_dim'], n_batches=parameters['n_batches']).to(device)
    generator = Autoencoder(input_shape=parameters['n_features'], latent_dim=parameters['latent_dim']).to(device)

    print('Discriminator:\n', discriminator)
    print('Number of parameters: ', discriminator.count_parameters())
    print('Generator:\n', generator)
    print('Number of parameters: ', generator.count_parameters())

    # create an optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=parameters['d_lr'])
    g_optimizer = optim.Adam(generator.parameters(), lr=parameters['g_lr'])

    # define losses
    d_criterion = nn.CrossEntropyLoss()
    g_criterion = nn.MSELoss()

    # split to values and batches
    data_batch_labels = data.iloc[:, 0]
    data_values = data.iloc[:, 1:]

    # get CV of benchmarks in original data
    cv_dict_original = batch_analysis.compute_cv_for_samples_types(data_values, benchmarks)

    # create and fit the scaler
    scaler = RobustScaler().fit(data_values)
    # apply scaling and do train test split
    X_train, X_test, y_train, y_test = preprocessing.split_to_train_and_test(data_values, data_batch_labels, scaler, proportion=parameters['train_ratio'])

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], shuffle=False)

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
    total_epochs = parameters['epochs']
    for epoch in range(total_epochs):

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
                g_loss -= parameters['d_lambda'] * d_loss.item()

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

        accuracy = numpy.mean(accuracy)  # for printing
        val_acc_history.append(accuracy)  # for plotting

        # collect variation coefficients for reg_types and benchmarks
        vcs = batch_analysis.compute_cv_for_samples_types(reconstruction, all_samples_types)
        reg_vcs_sum = 0.
        for sample in all_samples_types:
            if sample in reg_types:
                reg_vcs_sum += vcs[sample]  # calculate sum of (reg) variation coefs
            if sample in benchmarks:
                benchmarks_variation_coefs[sample].append(vcs[sample])

        reg_vc = reg_vcs_sum / len(vcs)  # compute mean overall variation coef
        reg_samples_vc_history.append(reg_vc)

        # assess cross correlations of regularization samples
        reg_corr = batch_analysis.get_sample_cross_correlation_estimate(reconstruction, reg_types, percent=25)
        reg_samples_corr_history.append(reg_corr)

        # assess cross correlations of benchmarks
        b_corr = batch_analysis.get_sample_cross_correlation_estimate(reconstruction, benchmarks)
        benchmarks_corr_history.append(b_corr)

        # collect clustering results for reg_types and benchmarks
        clustering, total_clusters = batch_analysis.compute_number_of_clusters_with_hdbscan(encodings, parameters, all_samples_types, print_info=False)
        # assess grouping of samples: compute g_lambda, so that it equals
        # 0, when all samples of a reg_type belong to the sample cluster
        # 1, when N samples of a reg_type belong to N different clusters
        b_grouping_coefs = []
        reg_grouping_coefs = []
        for sample in all_samples_types:
            n_sample_clusters = len(set(clustering[sample]))
            max_n_clusters = len(clustering[sample]) if len(clustering[sample]) <= total_clusters else total_clusters
            coef = (n_sample_clusters - 1) / max_n_clusters  # minus 1 to account for uncertainty in HDBSCAN
            if sample in reg_types:
                reg_grouping_coefs.append(coef)
            if sample in benchmarks:
                b_grouping_coefs.append(coef)

        b_grouping = numpy.mean(b_grouping_coefs)
        reg_grouping = numpy.mean(reg_grouping_coefs)

        benchmarks_grouping_history.append(b_grouping)
        reg_samples_grouping_history.append(reg_grouping)

        # SET REGULARIZATION FOR GENERATOR'S NEXT ITERATION
        g_regularizer = parameters['g_lambda'] * reg_grouping

        # SAVE MODEL
        torch.save(generator.state_dict(), save_to + '/checkpoints/ae_at_{}_{}.torch'.format(epoch, parameters['id']))

        # PRINT AND PLOT EPOCH INFO
        # plot every N epochs what happens with data
        if parameters['callback_step'] > 0 and epoch % parameters['callback_step'] == 0:
            # plot cross correlations of benchmarks in ALL reconstructed data
            batch_analysis.plot_batch_cross_correlations(reconstruction, 'epoch {}'.format(epoch+1), parameters['id'], benchmarks, save_to=save_to+'/callbacks/', save_plot=True)
            # plot umap of FULL encoded data
            batch_analysis.plot_encodings_umap(encodings, 'epoch {}'.format(epoch + 1), parameters, save_to=save_to + '/callbacks/')
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

    best_epoch = evaluation.find_best_epoch(history, skip_epochs=parameters['skip_epochs'])
    history.loc[best_epoch, 'best'] = True  # mark the best epoch

    evaluation.plot_losses(rec_loss_history, d_loss_history, g_loss_history, best_epoch, parameters, save_to=save_to)
    evaluation.plot_metrics(val_acc_history, reg_samples_corr_history, reg_samples_grouping_history, reg_samples_vc_history,
                 best_epoch, parameters, save_to=save_to)

    evaluation.plot_benchmarks_metrics(benchmarks_corr_history, benchmarks_grouping_history, best_epoch, parameters, save_to=save_to+'/benchmarks/')
    evaluation.plot_variation_coefs(benchmarks_variation_coefs, cv_dict_original, best_epoch, parameters, save_to=save_to+'/benchmarks/')

    # LOAD BEST MODEL
    generator = Autoencoder(input_shape=parameters['n_features'], latent_dim=parameters['latent_dim']).to(device)
    generator.load_state_dict(torch.load(save_to + 'checkpoints/ae_at_{}_{}.torch'.format(best_epoch, parameters['id']), map_location=device))
    generator.eval()

    # PLOT BEST EPOCH CALLBACKS
    scaled_data_values = scaler.transform(data_values.values)

    encodings = generator.encode(torch.Tensor(scaled_data_values))
    reconstruction = generator.decode(encodings)

    encodings = pandas.DataFrame(encodings.detach().numpy(), index=data_values.index)
    reconstruction = scaler.inverse_transform(reconstruction.detach().numpy())
    reconstruction = pandas.DataFrame(reconstruction, index=data_values.index, columns=data_values.columns)
    reconstruction = evaluation.mask_non_relevant_intensities(reconstruction, parameters['min_relevant_intensity'])

    # plot cross correlations of benchmarks in initial and normalized data
    batch_analysis.plot_batch_cross_correlations(data_values, 'initial data', parameters, benchmarks, save_to=save_to + '/benchmarks/', save_plot=True)
    batch_analysis.plot_batch_cross_correlations(reconstruction, 'at epoch {}'.format(best_epoch + 1), parameters, benchmarks, save_to=save_to+'/benchmarks/', save_plot=True)
    # plot umaps of initial, encoded and normalized data
    batch_analysis.plot_full_data_umaps(data_values, encodings, reconstruction, data_batch_labels, parameters, 'at epoch {}'.format(best_epoch + 1), save_to=save_to)
    pyplot.close('all')

    # SAVE ENCODED AND NORMALIZED DATA
    encodings.to_csv(save_to + 'encodings_{}.csv'.format(parameters['id']))
    reconstruction.T.to_csv(save_to + 'normalized_{}.csv'.format(parameters['id']))

    # SAVE PARAMETERS AND HISTORY
    pandas.DataFrame(parameters, index=['values'], columns=parameters.keys()).T.to_csv(save_to + 'parameters_{}.csv'.format(parameters['id']))
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