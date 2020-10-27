import pandas, numpy, os
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.constants import data_path as path
from src.preprocessing import get_fitted_scaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


if __name__ == '__main__':

    discriminator = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    discriminator.summary()

    latent_dim = 128

    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(7 * 7 * 128),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    # Instantiate one optimizer for the discriminator and another for the generator.
    d_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    g_optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


    @tf.function
    def train_step(real_images):

        print(real_images)

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        # Decode them to fake images
        generated_images = generator(random_latent_vectors)
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(labels.shape)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = discriminator(combined_images)
            d_loss = loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = discriminator(generator(random_latent_vectors))
            g_loss = loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        return d_loss, g_loss, generated_images

    # Prepare the dataset. We use both the training & test MNIST digits.
    batch_size = 64
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    all_digits = numpy.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = numpy.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    epochs = 100  # In practice you need at least 20 epochs to generate nice digits.
    save_dir = "../"

    for epoch in range(epochs):
        print("\nStart epoch", epoch)

        for step, real_images in enumerate(dataset):
            # Train the discriminator & generator on one batch of real images.
            d_loss, g_loss, generated_images = train_step(real_images)

            # Logging.
            if step % 200 == 0:
                # Print metrics
                print("discriminator loss at step %d: %.2f" % (step, d_loss))
                print("adversarial loss at step %d: %.2f" % (step, g_loss))

                # Save one generated image
                img = tf.keras.preprocessing.image.array_to_img(generated_images[0] * 255.0, scale=False)
                img.save(os.path.join(save_dir, "generated_img" + str(step) + ".png"))

            # To limit execution time we stop after 10 steps.
            # Remove the lines below to actually train the model!
            # if step > 10:
            #     break