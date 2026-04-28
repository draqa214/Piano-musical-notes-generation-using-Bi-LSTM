import keras
import numpy as np
from pre_process import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
LATENT_DIM = 128
LEARNING_RATE = 0.0002
NUM_UNITS = [256, 256]
EPOCHS = 50
BATCH_SIZE = 32
SAVE_GENERATOR_PATH = "model_generator.keras"
SAVE_DISCRIMINATOR_PATH = "model_discriminator.keras"


def build_generator(latent_dim, sequence_length, output_units):
    noise = keras.layers.Input(shape=(latent_dim,))

    x = keras.layers.Dense(sequence_length * 128, activation="relu")(noise)
    x = keras.layers.Reshape((sequence_length, 128))(x)
    x = keras.layers.LSTM(NUM_UNITS[0], return_sequences=True)(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(NUM_UNITS[1], return_sequences=True)(x)
    x = keras.layers.LayerNormalization()(x)
    output = keras.layers.TimeDistributed(
        keras.layers.Dense(output_units, activation="softmax")
    )(x)

    model = keras.Model(noise, output, name="generator")
    model.summary()
    return model


def build_discriminator(sequence_length, output_units, num_units):
    input = keras.layers.Input(shape=(sequence_length, output_units))

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(num_units[0], return_sequences=True)
    )(input)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(num_units[1]))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(input, output, name="discriminator")
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = keras.layers.Input(shape=(LATENT_DIM,))
    fake_sequence = generator(noise)
    validity = discriminator(fake_sequence)
    combined = keras.Model(noise, validity, name="gan")
    combined.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
    )
    return combined


def train_gan(
    output_units=OUTPUT_UNITS,
    num_units=NUM_UNITS,
    latent_dim=LATENT_DIM,
):
    # Load real sequences (inputs are one-hot encoded: shape (N, seq_len, vocab_size))
    real_sequences, _ = generate_training_sequences(SEQUENCE_LENGTH)
    n_real = len(real_sequences)

    generator = build_generator(latent_dim, SEQUENCE_LENGTH, output_units)
    discriminator = build_discriminator(SEQUENCE_LENGTH, output_units, num_units)
    gan = build_gan(generator, discriminator)

    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    for epoch in range(EPOCHS):
        # --- Train discriminator ---
        # Sample random real sequences
        idx = np.random.randint(0, n_real, BATCH_SIZE)
        real_batch = real_sequences[idx]

        # Generate fake sequences
        noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
        fake_batch = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_batch, fake_labels)
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])

        # --- Train generator (via combined model) ---
        noise = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
        # Generator wants discriminator to classify its output as real (label=1)
        g_loss = gan.train_on_batch(noise, real_labels)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"D loss: {d_loss:.4f}, D acc: {d_acc:.2%} | "
                f"G loss: {g_loss:.4f}"
            )

    generator.save(SAVE_GENERATOR_PATH)
    discriminator.save(SAVE_DISCRIMINATOR_PATH)
    print(f"Saved generator → {SAVE_GENERATOR_PATH}")
    print(f"Saved discriminator → {SAVE_DISCRIMINATOR_PATH}")


if __name__ == "__main__":
    train_gan()
