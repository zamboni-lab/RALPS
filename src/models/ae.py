import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from src.constants import data_path as path
from sklearn.preprocessing import StandardScaler, RobustScaler


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.e1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["latent_dim"])
        self.e2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d1 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["input_shape"])

        # best set of activations found
        self.e1_act = nn.CELU()
        self.e2_act = nn.Identity()
        self.d1_act = nn.CELU()
        self.d2_act = nn.Identity()

    def encode(self, features):
        encoded = self.e1(features)
        encoded = self.e1_act(encoded)
        encoded = self.e2(encoded)
        encoded = self.e2_act(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.d1(encoded)
        decoded = self.d1_act(decoded)
        decoded = self.d2(decoded)
        decoded = self.d2_act(decoded)
        return decoded

    def forward(self, features):
        encoded = self.encode(features)
        decoded = self.decode(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.e1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["latent_dim"])
        self.e2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d1 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["input_shape"])

        self.e1_act = kwargs['e1_act']
        self.e2_act = kwargs['e2_act']
        self.d1_act = kwargs['d1_act']
        self.d2_act = kwargs['d2_act']

    def encode(self, features):
        encoded = self.e1(features)
        encoded = self.e1_act(encoded)
        encoded = self.e2(encoded)
        encoded = self.e2_act(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.d1(encoded)
        decoded = self.d1_act(decoded)
        decoded = self.d2(decoded)
        decoded = self.d2_act(decoded)
        return decoded

    def forward(self, features):
        encoded = self.encode(features)
        decoded = self.decode(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data(shuffle=True):
    # collect merged dataset
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and remove metainfo
    data = data.iloc[:, 3:].T

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    if shuffle:
        data = data.sample(frac=1)

    return data


def scale_and_split_to_train_and_test(scaled_values, batches):

    n_samples, n_features = scaled_values.shape

    # split values to train and val
    x_train = scaled_values[:int(0.7 * n_samples), :]
    x_val = scaled_values[int(0.7 * n_samples):, :]
    y_train = batches[:int(0.7 * n_samples)]
    y_val = batches[int(0.7 * n_samples):]

    y_train -= 1
    y_val -= 1

    return x_train, x_val, y_train, y_val


def train_autoencoder_and_save_encodings():

    n_features = 170
    latent_dim = 50

    #  use gpu if available
    device = torch.device("cpu")

    model = Autoencoder(input_shape=n_features, latent_dim=latent_dim).to(device)

    print(model)
    print('Total number of parameters: ', model.count_parameters())

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # mean-squared error loss
    criterion = nn.L1Loss()

    # get data and do train test split
    data = get_data(shuffle=False)

    # split to values and batches
    batches = data.iloc[:, 0].values
    values = data.iloc[:, 1:].values

    # scale
    scaler = RobustScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(scaled))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)

    epochs = 150

    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reshape mini-batch data to [n_batches, n_features] matrix
            # load it to the active device
            batch_features = batch_features[0].to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # compute reconstructions
            outputs = model(batch_features)
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))

    print('saving model\n')
    saving_path = path.replace('data', 'res')
    torch.save(model.state_dict(), saving_path + 'autoencoder.models')

    print('encoding and saving full dataset')
    encoded = model.encode(torch.Tensor(scaled)).detach().numpy()

    # save encodings with corresponding batches for classification
    encodings = pandas.DataFrame(encoded, index=data.index)
    encodings.insert(0, 'batch', data.batch.values)
    encodings.to_csv(saving_path + 'encodings.csv')


def train_with_various_activations():
    """ Test different activation functions in the autoencoder. """

    n_features = 170
    latent_dim = 50
    epochs = 150
    device = torch.device("cpu")
    save_to = '/Users/dmitrav/ETH/projects/normalization/res/activations/'

    losses = [

        [nn.LeakyReLU(), nn.Identity(), nn.LeakyReLU(), nn.Identity()],
        [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        [nn.ReLU(), nn.Sigmoid(), nn.ReLU(), nn.Identity()],
        [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Identity()],

        [nn.ReLU(), nn.CELU(), nn.ReLU(), nn.CELU()],
        [nn.ReLU(), nn.Identity(), nn.ReLU(), nn.Identity()],
        [nn.CELU(), nn.Identity(), nn.CELU(), nn.Identity()],
        [nn.LeakyReLU(), nn.ReLU(), nn.ReLU(), nn.LeakyReLU()],
        [nn.ReLU(), nn.LeakyReLU(), nn.ReLU(), nn.LeakyReLU()],
        [nn.LeakyReLU(), nn.CELU(), nn.LeakyReLU(), nn.CELU()]
    ]

    # get data and do train test split
    data = get_data(shuffle=False)

    # split to values and batches
    batches = data.iloc[:, 0].values
    values = data.iloc[:, 1:].values

    # scale
    scaler = RobustScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    X_train = scaled

    for i, quad in enumerate(losses):

        model = TransformerAE(input_shape=n_features, latent_dim=latent_dim,
                              e1_act=quad[0], e2_act=quad[1], d1_act=quad[2], d2_act=quad[3]).to(device)

        print(model)
        print('Total number of parameters: ', model.count_parameters())

        # create an optimizer object
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # mean-squared error loss
        criterion = nn.L1Loss()

        # make datasets
        train_dataset = TensorDataset(torch.Tensor(X_train))

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)

        for epoch in range(epochs):
            loss = 0
            for batch_features in train_loader:
                # reshape mini-batch data to [n_batches, n_features] matrix
                # load it to the active device
                batch_features = batch_features[0].to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                # compute reconstructions
                outputs = model(batch_features)
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))

        # apply model of the last epoch to obtain reconstruction
        encodings = model.encode(torch.Tensor(scaled))
        reconstruction = model.decode(encodings)

        reconstruction = scaler.inverse_transform(reconstruction.detach().numpy())
        reconstruction = pandas.DataFrame(reconstruction, index=data.index).T
        reconstruction.to_csv(save_to + 'reconstruction_{}.csv'.format(i))


if __name__ == "__main__":
    train_autoencoder_and_save_encodings()