import torch, numpy, pandas, time
import torchvision as torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib import pyplot

from src.torch.cl import Classifier
from src.torch.ae import Autoencoder
from src.constants import samples_with_strong_batch_effects as benchmarks
from src.batch_analysis import compute_cv_for_samples_types, plot_batch_cross_correlations
from src.batch_analysis import compute_number_of_clusters_with_hdbscan, plot_full_dataset_umap


def split_to_train_and_test(values, batches, scaler):

    n_samples, n_features = values.shape

    # scale
    scaled = scaler.transform(values)

    # split values to train and val
    x_train = scaled[:int(0.7 * n_samples), :]
    x_val = scaled[int(0.7 * n_samples):, :]
    y_train = batches[:int(0.7 * n_samples)]
    y_val = batches[int(0.7 * n_samples):]

    y_train -= 1
    y_val -= 1

    return x_train, x_val, y_train, y_val


def get_data(path):
    # collect merged dataset
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and remove metainfo
    data = data.iloc[:, 3:].T

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)

    # collect encodings of the data by pretrained autoencoder
    encodings = pandas.read_csv(path.replace('data', 'res') + 'encodings.csv', index_col=0)

    return data, encodings


def plot_losses(d_loss, g_loss, val_acc, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(3, figsize=(6,9))

    fig.suptitle('Adversarial training loop')

    axs[0].plot(range(1, 1+len(d_loss)), d_loss)
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('CrossEntropy')
    axs[0].grid(True)

    axs[1].plot(range(1, 1+len(g_loss)), g_loss)
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('L1Loss - CrossEntropy')
    axs[1].grid(True)

    axs[2].plot(range(1, 1+len(val_acc)), val_acc)
    axs[2].set_title('Batch prediction')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Test accuracy')
    axs[2].grid(True)

    pyplot.tight_layout()
    pyplot.savefig(save_to + 'losses.pdf')


def plot_variation_coefs(vc_dict, vc_dict_original, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    if len(vc_dict) == 6:
        # save all on one figure
        pyplot.figure(figsize=(12, 8))
        for i, type in enumerate(vc_dict):

            x = range(1, 1+len(vc_dict[type]))  # epochs
            y = vc_dict[type]  # values

            ax = pyplot.subplot(2, 3, i + 1)
            ax.plot(x, y, label='Normalized data')
            ax.axhline(y=vc_dict_original[type], color='r', linestyle='-', label='Original data')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('VC')
            ax.set_title(type)
            ax.grid(True)
            ax.legend()

        pyplot.suptitle('Variation coefficients')
        pyplot.tight_layout()
        pyplot.savefig(save_to + 'vcs.pdf')
    else:
        # save one by one for each sample in dict
        for i, type in enumerate(vc_dict):
            x = range(1, 1+len(vc_dict[type]))  # epochs
            y = vc_dict[type]  # values

            pyplot.figure()
            pyplot.plot(x, y, label='Normalized data')
            pyplot.axhline(y=vc_dict_original[type], color='r', linestyle='-', label='Original data')
            pyplot.ylabel('VC')
            pyplot.xlabel('Epochs')
            pyplot.title('Variation coefficient for {}'.format(type))
            pyplot.grid(True)
            pyplot.tight_layout()
            pyplot.savefig(save_to + 'vcs_{}.pdf'.format(type))


def plot_n_clusters(clusters_dict, clusters_dict_original, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    if len(clusters_dict) == 6:
        # save all on one figure
        pyplot.figure(figsize=(12, 8))
        for i, type in enumerate(clusters_dict):

            x = range(1, 1+len(clusters_dict[type]))  # epochs
            y = clusters_dict[type]  # values

            ax = pyplot.subplot(2, 3, i + 1)
            ax.plot(x,y, label='Normalized data')
            ax.axhline(y=clusters_dict_original[type], color='r', linestyle='-', label='Original data')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Clusters found')
            ax.set_title(type)
            ax.grid(True)
            ax.legend()

        pyplot.suptitle('HDBSCAN clustering')
        pyplot.tight_layout()
        pyplot.savefig(save_to + 'clustering.pdf')
    else:
        # save one by one for each sample in dict
        for i, type in enumerate(clusters_dict):
            x = range(1, 1+len(clusters_dict[type]))  # epochs
            y = clusters_dict[type]  # values

            pyplot.figure()
            pyplot.plot(x, y, label='Normalized data')
            pyplot.axhline(y=clusters_dict_original[type], color='r', linestyle='-', label='Original data')
            pyplot.ylabel('Clusters found')
            pyplot.xlabel('Epochs')
            pyplot.title('HDBSCAN clustering for {}'.format(type))
            pyplot.grid(True)
            pyplot.tight_layout()
            pyplot.legend()
            pyplot.savefig(save_to + 'clustering_{}.pdf'.format(type))


if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/data/'
    save_to = path.replace('data', 'res')
    n_features = 170
    latent_dim = 100
    n_batches = 7

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create models
    discriminator = Classifier(input_shape=latent_dim, n_batches=n_batches).to(device)
    generator = Autoencoder(input_shape=n_features, latent_dim=latent_dim).to(device)

    print('Discriminator:\n', discriminator)
    print('Number of parameters: ', discriminator.count_parameters())
    print('Generator:\n', generator)
    print('Total number of parameters: ', generator.count_parameters())

    # create an optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-3)
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)

    # define losses
    d_criterion = nn.CrossEntropyLoss()
    g_criterion = nn.L1Loss()
    # g_criterion = nn.MSELoss()

    data, pretrained_encodings = get_data(path)
    # split to values and batches
    data_batch_labels = data.iloc[:, 0]
    data_values = data.iloc[:, 1:]

    # get CV of benchmarks in original data
    cv_dict_original = compute_cv_for_samples_types(data_values, sample_types_of_interest=benchmarks)
    clustering_dict_original = compute_number_of_clusters_with_hdbscan(pretrained_encodings, print_info=False, sample_types_of_interest=benchmarks)

    # create and fit the scaler
    scaler = RobustScaler().fit(data_values)
    # apply scaling and do train test split
    X_train, X_test, y_train, y_test = split_to_train_and_test(data_values, data_batch_labels, scaler)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    epochs = 200

    # Lists to keep track of progress
    g_loss_history = []
    d_loss_history = []
    val_acc_history = []
    variation_coefs = dict([(sample, []) for sample in benchmarks])
    n_clusters = dict([(sample, []) for sample in benchmarks])

    for epoch in range(epochs+1):
        start = time.time()
        d_loss_per_epoch = 0
        g_loss_per_epoch = 0
        total_loss_per_epoch = 0
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
            # compute reconstructions
            reconstruction = generator(batch_features)
            # compute training reconstruction loss
            g_loss = g_criterion(reconstruction, batch_features)
            # compute accumulated gradients
            g_loss.backward()
            g_loss_per_epoch += g_loss.item()

            # substitute discriminator loss to push it towards smaller batch effects
            g_loss -= d_loss
            total_loss_per_epoch += g_loss.item()

            # perform parameter update based on current gradients
            g_optimizer.step()

        # COMPUTE EPOCH LOSSES
        d_loss = d_loss_per_epoch / len(train_loader)
        g_loss = g_loss_per_epoch / len(train_loader)

        d_loss_history.append(d_loss)
        g_loss_history.append(g_loss)

        # GENERATE DATA FOR OTHER METRICS
        encodings = generator.encode(torch.Tensor(data_values.values))
        reconstruction = generator.decode(encodings)

        encodings = pandas.DataFrame(encodings.detach().numpy(), index=data_values.index)
        encodings.insert(0, 'batch', data_batch_labels)
        reconstruction = pandas.DataFrame(reconstruction.detach().numpy(), index=data_values.index)

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
        vc = compute_cv_for_samples_types(reconstruction, sample_types_of_interest=benchmarks)
        for sample in benchmarks:
            variation_coefs[sample].append(vc[sample])

        # collect clustering results for some samples of ALL encoded data
        clustering = compute_number_of_clusters_with_hdbscan(encodings, print_info=False, sample_types_of_interest=benchmarks)
        for sample in benchmarks:
            n_clusters[sample].append(clustering[sample])

        # PRINT AND PLOT EPOCH INFO
        # plot every N epochs what happens with data
        if epoch % 10 == 0:
            # assess cross correlations of benchmarks in ALL reconstructed data
            plot_batch_cross_correlations(reconstruction, 'epoch {}'.format(epoch+1), sample_types_of_interest=benchmarks, save_to=save_to+'callbacks/')
            # plot umap of FULL encoded data
            plot_full_dataset_umap(encodings, 'epoch {}'.format(epoch+1), sample_types_of_interest=benchmarks, save_to=save_to+'callbacks/')

        # display the epoch training loss
        timing = int(time.time() - start)
        print("epoch {}/{}, {} seconds elapsed: d_loss = {:.4f}, g_loss = {:.4f}, val_acc = {:.4f}".format(epoch + 1, epochs, timing, d_loss, g_loss, accuracy))

    # PLOT TRAINING HISTORY
    plot_losses(d_loss_history, g_loss_history, val_acc_history, save_to=save_to+'callbacks/')
    plot_variation_coefs(variation_coefs, cv_dict_original, save_to=save_to+'callbacks/')
    plot_n_clusters(n_clusters, clustering_dict_original, save_to=save_to+'callbacks/')

    # TODO: ideas:
    #    i) experiment with coefficient at g_loss -= d_loss
    #  iii) try adding regularizations to g_loss and d_loss
    #   iv) come up with a criterion: what does it mean exactly to remove batch effects in this case? is it generalizable?

