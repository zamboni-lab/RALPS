import torch, numpy, pandas
import torchvision as torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib import pyplot

from src.torch.cl import Classifier
from src.torch.ae import Autoencoder
from src.constants import samples_with_strong_batch_effects as benchmarks
from src.batch_analysis import compute_cv_for_samples_types, plot_batch_cross_correlations
from src.batch_analysis import plot_batch_effects_with_umap, compute_number_of_clusters_with_hdbscan


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

    return data


def plot_losses(d_loss, g_loss, val_acc, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    fig, axs = pyplot.subplots(3)

    fig.suptitle('Adversarial training loop')

    axs[0].plot(range(epochs), d_loss)
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('CrossEntropy')
    axs[0].grid()

    axs[1].plot(range(epochs), g_loss)
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('L1Loss - CrossEntropy')
    axs[1].grid()

    axs[2].plot(range(epochs), val_acc)
    axs[2].set_title('Batch prediction')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Test accuracy')
    axs[2].grid()

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(save_to + 'losses.pdf')


def plot_variation_coefs(vc_dict, save_to='/Users/andreidm/ETH/projects/normalization/res/'):

    # pyplot.figure(figsize=(12, 8))
    #
    # for i, type in enumerate(samples_by_types):
    #     df = data.loc[:, numpy.array(samples_by_types[type])]
    #     df.columns = [x[-6:] for x in df.columns]
    #     df = df.corr()
    #
    #     ax = pyplot.subplot(2, 3, i + 1)
    #     seaborn.heatmap(df)
    #     ax.set_title(type)
    #
    # pyplot.suptitle('Cross correlations: {}'.format(method_name))
    # pyplot.tight_layout()
    # # pyplot.show()
    # pyplot.savefig(save_to + 'correlations_{}.pdf'.format(method_name.replace(' ', '_')))

    pass


def plot_n_clusters(clusters_dict, save_to='/Users/andreidm/ETH/projects/normalization/res/'):
    pass


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
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)

    # define losses
    d_criterion = nn.CrossEntropyLoss()
    g_criterion = nn.L1Loss()
    # g_criterion = nn.MSELoss()

    data = get_data(path)
    # split to values and batches
    data_batch_labels = data.iloc[:, 0]
    data_values = data.iloc[:, 1:]

    # create and fit the scaler
    scaler = RobustScaler().fit(data_values)
    # apply scaling and do train test split
    X_train, X_test, y_train, y_test = split_to_train_and_test(data_values, data_batch_labels, scaler)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    epochs = 300

    # Lists to keep track of progress
    g_loss_history = []
    d_loss_history = []
    val_acc_history = []
    variation_coefs = dict([(sample, []) for sample in benchmarks])
    n_clusters = dict([(sample, []) for sample in benchmarks])

    for epoch in range(epochs):
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

        # display the epoch training loss
        print("epoch : {}/{}, d_loss = {:.4f}, g_loss = {:.4f}, val_acc = {:.4f} ".format(epoch + 1, epochs, d_loss, g_loss, accuracy))

        # GENERATE DATA FOR OTHER METRICS
        reconstruction = generator(torch.Tensor(data_values)).detach().numpy()
        reconstruction = pandas.DataFrame(reconstruction, columns=data_values.columns)

        encodings = generator.encode(torch.Tensor(data_values)).detach().numpy()
        encodings = pandas.DataFrame(encodings, index=data_values.index)
        encodings.insert(0, 'batch', data_batch_labels)

        # COMPUTE METRICS
        # classification accuracy of TEST data
        accuracy = []
        for batch_features, labels in test_loader:

            batch_features = batch_features.to(device)
            # generate encodings
            encodings = generator.encode(batch_features)

            predictions = discriminator(encodings)

            # calculate accuracy per batch
            true_positives = (predictions.argmax(-1) == labels).float().detach().numpy()
            batch_accuracy = true_positives.sum() / len(true_positives)
            accuracy.append(batch_accuracy)

        val_acc_history.append(numpy.mean(accuracy))

        # collect variation coefficients for some samples of ALL reconstructed data
        vc = compute_cv_for_samples_types(reconstruction, sample_types_of_interest=benchmarks)
        for sample in benchmarks:
            variation_coefs[sample].append(vc[sample])

        # collect clustering results for some samples of ALL encoded data
        clustering = compute_number_of_clusters_with_hdbscan(encodings, print_info=False, sample_types_of_interest=benchmarks)
        for sample in benchmarks:
            n_clusters[sample].append(clustering[sample])

        if epoch % 10 > 0:
            # assess cross correlations of benchmarks in ALL reconstructed data
            plot_batch_cross_correlations(reconstruction, 'epoch {}'.format(epoch), sample_types_of_interest=benchmarks, save_to=path+'callbacks/')
            # assess batch effects in benchmarks in ALL encoded data
            plot_batch_effects_with_umap(encodings, 'epoch {}'.format(epoch), sample_types_of_interest=benchmarks, save_to=path+'callbacks/')

    # PLOT TRAINING HISTORY
    plot_losses(d_loss_history, g_loss_history, val_acc_history, save_to=path+'callbacks/')
    plot_variation_coefs(variation_coefs, save_to=path+'callbacks/')
    plot_n_clusters(n_clusters, save_to=path+'callbacks/')


    # TODO: ideas:
    #   ii) plot umaps of encodings before and after training (better every n epochs)
    #    i) experiment with coefficient at g_loss -= d_loss
    #  iii) try adding regularizations to g_loss and d_loss
    #   iv) come up with a criterion: what does it mean exactly to remove batch effects in this case? is it generalizable?

