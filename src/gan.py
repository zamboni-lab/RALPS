import pandas, numpy
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.constants import data_path as path
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class GAN(keras.Model):

    def __init__(self, discriminator, generator, scaler):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.autoencoder = generator['autoencoder']
        self.encoder = generator['encoder']
        self.decoder = generator['decoder']
        self.scaler = scaler

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):

        X, y = data

        print(X.shape[0])

        X = self.scaler.transform(X)
        X_encoded = self.encoder.predict(X)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(X_encoded)
            d_loss = keras.losses.CategoricalCrossentropy(y, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        X_restored = self.autoencoder.predict(X)

        # Train the generator (note that we should NOT update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            ae_loss = keras.losses.MeanAbsoluteError(X, X_restored)
            reg_loss = 0.  # opportunity to regularize
            g_loss = sum([ae_loss, -d_loss, reg_loss])

        grads = tape.gradient(g_loss, self.autoencoder.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


def get_models():

    latent_dim, n_features, n_batches = 100, 170, 7

    # Create the discriminator
    discriminator = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(7, activation='relu'),
        layers.Dense(n_batches, activation='softmax')
    ], name="discriminator")

    # Create the generator

    autoencoder = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(latent_dim, activation='relu'),
        layers.Dense(n_features, activation='tanh')
    ], name="autoencoder")

    encoder = keras.Sequential([autoencoder.layers[0], autoencoder.layers[1]], name="encoder")
    decoder = keras.Sequential([keras.Input(shape=(latent_dim,)), autoencoder.layers[-1]], name="decoder")

    generator = {'autoencoder': autoencoder, 'encoder': encoder, 'decoder': decoder}

    return generator, discriminator


if __name__ == "__main__":

    # collect merged dataset
    data = pandas.read_csv(path + 'filtered_data.csv')
    batch_info = pandas.read_csv(path + 'batch_info.csv')

    # transpose and remove metainfo
    data = data.iloc[:, 3:].T

    # add batch and shuffle
    data.insert(0, 'batch', batch_info['batch'].values)
    data = data.sample(frac=1)

    # split to values and batches
    Y = data.iloc[:, 0].values - 1
    Y = to_categorical(Y)
    n_batches = Y.shape[1]
    X = data.iloc[:, 1:].values
    n_samples, n_features = X.shape

    # fit scaler
    scaler = RobustScaler().fit(X)

    # split into train and val
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    numpy.reshape(X_train, (-1, n_features, 1))
    numpy.reshape(X_val, (-1, n_features, 1))
    numpy.reshape(y_train, (-1, n_batches, 1))
    numpy.reshape(y_val, (-1, n_batches, 1))


    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    # specify models
    generator, discriminator = get_models()

    gan = GAN(discriminator=discriminator, generator=generator, scaler=scaler)

    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003)
    )

    gan.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_val, y_val))