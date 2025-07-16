import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os
import logging
from typing import Tuple, List
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)

        # Encoder
        self.down_stack = [
            self._downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self._downsample(128, 4),  # (bs, 64, 64, 128)
            self._downsample(256, 4),  # (bs, 32, 32, 256)
            self._downsample(512, 4),  # (bs, 16, 16, 512)
            self._downsample(512, 4),  # (bs, 8, 8, 512)
            self._downsample(512, 4),  # (bs, 4, 4, 512)
            self._downsample(512, 4),  # (bs, 2, 2, 512)
            self._downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        # Decoder
        self.up_stack = [
            self._upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self._upsample(512, 4),  # (bs, 16, 16, 1024)
            self._upsample(256, 4),  # (bs, 32, 32, 512)
            self._upsample(128, 4),  # (bs, 64, 64, 256)
            self._upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        self.last = Conv2DTranspose(3, 4,
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=self.initializer,
                                    activation='tanh')  # (bs, 256, 256, 3)

    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = self.initializer

        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(BatchNormalization())

        result.add(LeakyReLU())
        return result

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = self.initializer

        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        result.add(BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

    def call(self, x, training=True):
        skips = []

        # Encoder
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Decoder
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)

        # Define layers
        self.conv1 = Conv2D(64, 4, strides=2, padding='same',
                            kernel_initializer=self.initializer)
        self.conv2 = Conv2D(128, 4, strides=2, padding='same',
                            kernel_initializer=self.initializer)
        self.conv3 = Conv2D(256, 4, strides=2, padding='same',
                            kernel_initializer=self.initializer)
        self.conv4 = Conv2D(512, 4, strides=2, padding='same',
                            kernel_initializer=self.initializer)
        self.output_layer = Conv2D(1, 4, padding='same',
                                   kernel_initializer=self.initializer)

        # Batch normalization layers
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, inputs, training=True):
        # Ensure inputs is a single tensor
        if isinstance(inputs, (list, tuple)):
            x = tf.concat(inputs, axis=-1)
        else:
            x = inputs

        x = self.conv1(x)
        x = LeakyReLU(0.2)(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = LeakyReLU(0.2)(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = LeakyReLU(0.2)(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = LeakyReLU(0.2)(x)

        return self.output_layer(x)


class GANTrainer:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.LAMBDA = 100

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss = self._generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self._discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def _generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output),
                                                                        disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
        return total_gen_loss

    def _discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output),
                                                                         disc_real_output)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output),
                                                                              disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def load_dataset(image_path: str, target_path: str, batch_size: int = 1) -> tf.data.Dataset:
        def load_image(image_file):
            image = tf.io.read_file(image_file)
            image = tf.image.decode_jpeg(image)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [256, 256])
            image = (image / 127.5) - 1
            return image

        input_images = tf.data.Dataset.list_files(str(image_path + "/*.jpg"))
        target_images = tf.data.Dataset.list_files(str(target_path + "/*.jpg"))

        input_dataset = input_images.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        target_dataset = target_images.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

        BUFFER_SIZE = 1000

        return train_dataset.shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def display_sample(generator: Generator, test_input: tf.Tensor, test_target: tf.Tensor):
        generated_image = generator(test_input, training=False)
        plt.figure(figsize=(12, 12))

        images = [test_input[0], test_target[0], generated_image[0]]
        titles = ['Input', 'Target', 'Generated']

        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 3, i + 1)
            plt.title(title)
            plt.imshow(image * 0.5 + 0.5)  # Denormalize the image
            plt.axis('off')

        plt.show()

    def main():
        # Set hyperparameters
        BATCH_SIZE = 1
        EPOCHS = 50

        # Set paths
        image_path = '/content/trainA/'
        target_path = '/content/trainB/'

        # Load dataset
        train_dataset = load_dataset(image_path, target_path, BATCH_SIZE)

        # Create models
        generator = Generator()
        discriminator = Discriminator()

        # Create trainer
        trainer = GANTrainer(generator, discriminator)

        # Training loop
        for epoch in range(EPOCHS):
            logging.info(f"Epoch {epoch + 1}/{EPOCHS}")

            total_gen_loss = 0
            total_disc_loss = 0
            num_batches = 0

            for input_image, target in train_dataset:
                gen_loss, disc_loss = trainer.train_step(input_image, target)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss
                num_batches += 1

                if num_batches % 50 == 0:
                    logging.info(f"Batch {num_batches}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

            avg_gen_loss = total_gen_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
            logging.info(f"Generator Loss: {avg_gen_loss:.4f}")
            logging.info(f"Discriminator Loss: {avg_disc_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                for input_image, target in train_dataset.take(1):
                    display_sample(generator, input_image, target)
                    break

        # Save the trained generator
        generator.save_weights('generator_weights.h5')
        logging.info("Training completed. Generator weights saved as 'generator_weights.h5'")

    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise
