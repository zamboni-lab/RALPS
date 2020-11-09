import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from constants import data_path as path
from sklearn.preprocessing import StandardScaler, RobustScaler


class Autoencoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.e1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["latent_dim"])
        self.e2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d1 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["latent_dim"])
        self.d2 = nn.Linear(in_features=kwargs["latent_dim"], out_features=kwargs["input_shape"])

    def encode(self, features):
        encoded = self.e1(features)
        encoded = nn.LeakyReLU()(encoded)
        encoded = self.e2(encoded)
        encoded = nn.CELU()(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.d1(encoded)
        decoded = nn.LeakyReLU()(decoded)
        decoded = self.d2(decoded)
        decoded = nn.CELU()(decoded)
        return decoded

    def forward(self, features):
        encoded = self.e1(features)
        encoded = nn.LeakyReLU()(encoded)
        encoded = self.e2(encoded)
        encoded = nn.CELU()(encoded)
        decoded = self.d1(encoded)
        decoded = nn.LeakyReLU()(decoded)
        decoded = self.d2(decoded)
        decoded = nn.CELU()(decoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data():
    # collect merged dataset
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and remove metainfo
    data = data.iloc[:, 3:].T

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)

    return data


def split_to_train_and_test(data):

    # split to values and batches
    batches = data.iloc[:, 0].values
    values = data.iloc[:, 1:].values
    n_samples, n_features = values.shape

    # scale
    scaler = RobustScaler()
    scaled = scaler.fit_transform(values)

    # split values to train and val
    x_train = scaled[:int(0.7 * n_samples), :]
    x_val = scaled[int(0.7 * n_samples):, :]
    y_train = batches[:int(0.7 * n_samples)]
    y_val = batches[int(0.7 * n_samples):]

    y_train -= 1
    y_val -= 1

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":

    n_features = 170
    latent_dim = 100

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = Autoencoder(input_shape=n_features, latent_dim=latent_dim).to(device)

    print(model)
    print('Total number of parameters: ', model.count_parameters())

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    # get data and do train test split
    data = get_data()
    X_train, X_test, y_train, y_test = split_to_train_and_test(data)

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train))
    test_dataset = TensorDataset(torch.Tensor(X_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    epochs = 200

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

        test_loss = 0
        for batch_features in test_loader:
            batch_features = batch_features[0].to(device)
            outputs = model(batch_features)
            test_loss += criterion(outputs, batch_features).item()

        # compute the epoch training loss
        loss = loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, test_loss = {:.6f}".format(epoch + 1, epochs, loss, test_loss))

    print('saving model\n')
    saving_path = path.replace('data', 'res')
    torch.save(model.state_dict(), path + 'autoencoder.models')

    print('encoding and saving full dataset')
    X = numpy.concatenate([X_train, X_test])
    X_encoded = model.encode(torch.Tensor(X)).detach().numpy()
    # save encodings with corresponding batches for classification
    encodings = pandas.DataFrame(X_encoded, index=data.index)
    encodings.insert(0, 'batch', data.batch.values)
    encodings.to_csv(path.replace('data', 'res') + 'encodings.csv')
