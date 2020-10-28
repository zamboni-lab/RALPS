import torch, numpy, pandas
import torchvision as torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib import pyplot

from src.torch.cl import Classifier
from src.torch.ae import Autoencoder


def split_to_train_and_test(data, scaler):
    # split to values and batches
    batches = data.iloc[:, 0].values
    values = data.iloc[:, 1:].values
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


if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/data/'
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
    # create and fit the scaler
    scaler = RobustScaler().fit(data.iloc[:, 1:].values)
    # apply scaling and do train test split
    X_train, X_test, y_train, y_test = split_to_train_and_test(data, scaler)

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

        # collect metrics
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

        # compute epoch losses
        d_loss = d_loss_per_epoch / len(train_loader)
        g_loss = g_loss_per_epoch / len(train_loader)
        accuracy = numpy.mean(accuracy)

        d_loss_history.append(d_loss)
        g_loss_history.append(g_loss)
        val_acc_history.append(accuracy)

        # display the epoch training loss
        print("epoch : {}/{}, d_loss = {:.4f}, g_loss = {:.4f}, val_acc = {:.4f} ".format(epoch + 1, epochs, d_loss, g_loss, accuracy))

    # PLOT TRAINING HISTORY
    fig, axs = pyplot.subplots(3)

    fig.suptitle('Adversarial training loop')

    axs[0].plot(range(epochs), d_loss_history)
    axs[0].set_title('Classifier loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('CrossEntropy')
    axs[0].grid()

    axs[1].plot(range(epochs), g_loss_history)
    axs[1].set_title('Autoencoder loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('L1Loss - CrossEntropy')
    axs[1].grid()

    axs[2].plot(range(epochs), val_acc_history)
    axs[2].set_title('Batch prediction')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Test accuracy')
    axs[2].grid()

    pyplot.tight_layout()
    pyplot.show()

