import pandas, numpy
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.constants import data_path as path
from src.preprocessing import get_fitted_scaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


class GAN(keras.Model):

    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.autoencoder = generator['autoencoder']
        self.encoder = generator['encoder']
        self.decoder = generator['decoder']

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):

        X, y = data

        print(X)

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


class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_freatures = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels


if __name__ == "__main__":

    # train_dataset = tf.data.experimental.make_csv_dataset(path + 'train.csv', batch_size=128, num_epochs=1, label_name='batch')
    # test_dataset = tf.data.experimental.make_csv_dataset(path + 'test.csv', batch_size=128, num_epochs=1, label_name='batch')
    #
    # NUMERIC_FEATURES = [str(x) for x in range(170)]
    # train_packed = train_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
    # test_packed = test_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
    #
    # features, labels = next(iter(train_packed))

    train_data = pandas.read_csv(path + 'train.csv', index_col=False).values
    test_data = pandas.read_csv(path + 'test.csv', index_col=False).values

    X_train = train_data[:, 1:].astype('float32')
    X_test = test_data[:, 1:].astype('float32')
    y_train = train_data[:, 0]
    y_test = test_data[:, 0]

    scaler = get_fitted_scaler()

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = numpy.reshape(X_train, (-1, 170, 1))
    X_test = numpy.reshape(X_test, (-1, 170, 1))
    y_train = numpy.reshape(y_train, (-1, 1, 1))
    y_test = numpy.reshape(y_test, (-1, 1, 1))

    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)

    train_dataset = tf.data.Dataset.zip((X_train, y_train)).batch(64)
    test_dataset = tf.data.Dataset.zip((X_test, y_test)).batch(64)

    # specify models
    generator, discriminator = get_models()

    gan = GAN(discriminator=discriminator, generator=generator)

    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003)
    )

    gan.fit(train_dataset, epochs=1, validation_data=test_dataset)