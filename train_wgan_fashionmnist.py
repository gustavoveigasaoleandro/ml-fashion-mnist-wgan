import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

OUTPUT_DIR = Path("fashionmnist_wgan_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EPOCHS = 30
NOISE_DIMENSION = 100
NUM_IMAGES = 16
BATCH_SIZE = 512
CHECKPOINT_DIR = OUTPUT_DIR / "training_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def build_generator() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Input(shape=(NOISE_DIMENSION,)),
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, (5, 5), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )


def build_discriminator() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ]
    )


def loss_generator(fake_output: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_output)


def loss_discriminator(
    real_output: tf.Tensor, fake_output: tf.Tensor, gradient_penalty_value: tf.Tensor
) -> tf.Tensor:
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gradient_penalty_value


def create_and_save_images(model: tf.keras.Model, epoch: int, test_input: tf.Tensor) -> None:
    predictions = model(test_input, training=False).numpy()
    figure = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"img_epoch_{epoch:04d}.png", dpi=150)
    plt.close(figure)


def save_loss_curve(generator_losses: list[float], discriminator_losses: list[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(generator_losses, label="Gerador")
    plt.plot(discriminator_losses, label="Discriminador")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "losses.png", dpi=150)
    plt.close()


def main() -> None:
    (x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32")
    x_train = (x_train - 127.5) / 127.5

    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(x_train.shape[0])
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    generator = build_generator()
    discriminator = build_discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.5, beta_2=0.9
    )
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    @tf.function
    def gradient_penalty(real_images: tf.Tensor, fake_images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(real_images)[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_images = epsilon * real_images + (1.0 - epsilon) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            scores = discriminator(interpolated_images, training=True)

        gradients = gp_tape.gradient(scores, interpolated_images)
        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)
        return tf.reduce_mean((gradient_norm - 1.0) ** 2)

    @tf.function
    def training_step(images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(images)[0]

        for _ in tf.range(2):
            noise = tf.random.normal([batch_size, NOISE_DIMENSION])

            with tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)
                gp_value = gradient_penalty(images, generated_images)
                discriminator_loss = loss_discriminator(real_output, fake_output, gp_value)

            discriminator_gradients = disc_tape.gradient(
                discriminator_loss, discriminator.trainable_variables
            )
            discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, discriminator.trainable_variables)
            )

        noise = tf.random.normal([batch_size, NOISE_DIMENSION])
        with tf.GradientTape() as gen_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            generator_loss = loss_generator(fake_output)

        generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        return generator_loss, discriminator_loss

    seed = tf.random.normal([NUM_IMAGES, NOISE_DIMENSION])
    history_g: list[float] = []
    history_d: list[float] = []

    for epoch in range(EPOCHS):
        start = time.time()
        epoch_g = []
        epoch_d = []

        for image_batch in dataset:
            generator_loss, discriminator_loss = training_step(image_batch)
            epoch_g.append(float(generator_loss.numpy()))
            epoch_d.append(float(discriminator_loss.numpy()))

        history_g.append(float(np.mean(epoch_g)))
        history_d.append(float(np.mean(epoch_d)))
        create_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=str(CHECKPOINT_DIR / "ckpt"))

        print(
            f"Epoca {epoch + 1:03d} | "
            f"loss gerador: {history_g[-1]:.4f} | "
            f"loss discriminador: {history_d[-1]:.4f} | "
            f"tempo: {time.time() - start:.2f}s"
        )

    generator.save(OUTPUT_DIR / "generator_fashionmnist.h5")
    save_loss_curve(history_g, history_d)
    summary = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "noise_dimension": NOISE_DIMENSION,
        "generator_loss_last": history_g[-1],
        "discriminator_loss_last": history_d[-1],
        "generated_image_examples": sorted(p.name for p in OUTPUT_DIR.glob("img_epoch_*.png"))[-5:],
    }
    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
