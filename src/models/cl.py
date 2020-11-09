import torch, numpy, pandas
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from constants import data_path as path
from sklearn.preprocessing import StandardScaler, RobustScaler


class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.l1 = nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["n_batches"])
        self.l2 = nn.Linear(in_features=kwargs["n_batches"], out_features=kwargs["n_batches"])

    def forward(self, x):
        x = self.l1(x)
        x = nn.LeakyReLU()(x)
        x = self.l2(x)
        x = nn.Softmax(dim=1)(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data(path):

    encodings = pandas.read_csv(path, index_col=0)

    # split to values and batches
    batches = encodings.iloc[:, 0].values
    values = encodings.iloc[:, 1:].values
    n_samples, n_features = values.shape

    # split values to train and val
    x_train = values[:int(0.7 * n_samples), :]
    x_val = values[int(0.7 * n_samples):, :]
    y_train = batches[:int(0.7 * n_samples)]
    y_val = batches[int(0.7 * n_samples):]

    y_train -= 1
    y_val -= 1

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":

    path = '/Users/andreidm/ETH/projects/normalization/res/'
    latent_dim = 100
    n_batches = 7

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = Classifier(input_shape=latent_dim, n_batches=n_batches).to(device)

    print(model)
    print('Total number of parameters: ', model.count_parameters())

    # create an optimizer object
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.CrossEntropyLoss()

    X_train, X_test, y_train, y_test = get_data(path + 'encodings.csv')

    # make datasets
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    epochs = 20

    for epoch in range(epochs):
        loss = 0
        for batch_features, labels in train_loader:
            # reshape mini-batch data to [n_batches, n_features] matrix
            # load it to the active device
            batch_features = batch_features.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # predict batches
            outputs = model(batch_features)
            # compute training classification loss
            train_loss = criterion(outputs, labels)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        test_loss = 0
        accuracy = []
        for batch_features, labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            test_loss += criterion(outputs, labels).item()

            # calculate accuracy per batch
            true_positives = (outputs.argmax(-1) == labels).float().detach().numpy()
            batch_accuracy = true_positives.sum() / len(true_positives)
            accuracy.append(batch_accuracy)

        # compute epoch losses
        loss = loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        # compute accuracy metric
        test_acc = numpy.mean(accuracy)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, test_loss = {:.6f}, test_acc = {:.6f}".format(epoch + 1, epochs, loss, test_loss, test_acc))

    predictions = model(torch.Tensor(X_test))
    true_positives = (predictions.argmax(-1) == torch.LongTensor(y_test)).float().detach().numpy()
    test_accuracy = true_positives.sum() / len(true_positives)
    print('achieved accuracy:', test_accuracy)

    print('saving model\n')
    torch.save(model.state_dict(), path + 'classifier.models')

    print('creating new model')
    model = Classifier(input_shape=latent_dim, n_batches=n_batches)
    print('loading state_dict()')
    model.load_state_dict(torch.load(path + 'classifier.models', map_location=device))
    model.eval()

    predictions = model(torch.Tensor(X_test))
    true_positives = (predictions.argmax(-1) == torch.LongTensor(y_test)).float().detach().numpy()
    test_accuracy = true_positives.sum() / len(true_positives)
    print('achieved accuracy:', test_accuracy)


    print()