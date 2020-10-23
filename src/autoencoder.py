
import pandas, numpy
from src import preprocessing

import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":

    # collect merged dataset
    path = '/Users/andreidm/ETH/projects/normalization/data/'
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and filter
    data = data.iloc[:, 3:].T
    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)
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

    latent_dim = 100

    input = tf.keras.Input(shape=(n_features,))
    encoded = layers.Dense(latent_dim, activation='relu')(input)
    decoded = layers.Dense(n_features, activation='tanh')(encoded)

    autoencoder = tf.keras.Model(input, decoded)
    encoder = tf.keras.Model(input, encoded)

    encoded_input = tf.keras.Input(shape=(latent_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = tf.keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
    autoencoder.summary()

    # 2000 epochs -> loss: 51.7836 - val_loss: 52.1401
    autoencoder.fit(x_train, x_train,
                    epochs=2000,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_val, x_val))

    # val_encoded = encoder.predict(x_val)
    # val_decoded = decoder.predict(val_encoded)
    # val_before = scaler.inverse_transform(x_val)
    # val_after = scaler.inverse_transform(val_decoded)

    scaled_encoded = encoder.predict(scaled)
    encodings = pandas.DataFrame(scaled_encoded, index=data.index)
    encodings.insert(0, 'batch', batches)

    encodings.to_csv(path.replace('data', 'res') + 'samples_encodings.csv')

    print()