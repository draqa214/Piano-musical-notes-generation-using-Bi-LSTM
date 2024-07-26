import keras
from pre_process import generate_training_sequences, SEQUENCE_LENGTH
import numpy as np

OUTPUT_UNITS = 38
LOSS = "binary_crossentropy"  # Binary cross-entropy for binary classification
LEARNING_RATE = 0.001
NUM_UNITS = [256, 256]  # Increase the number of units
EPOCHS = 5
BATCH_SIZE = 16
SAVE_MODEL_PATH = "model_discriminator.h5"


def build_discriminator(output_units, num_units, loss, learning_rate):
    # create model architecture
    input = keras.layers.Input(shape=(None, output_units))

    # Bidirectional LSTM layers
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(num_units[0], return_sequences=True)
    )(input)
    x = keras.layers.Dropout(0.2)(x)

    # Another Bidirectional LSTM layer
    x = keras.layers.Bidirectional(keras.layers.LSTM(num_units[1]))(x)
    x = keras.layers.Dropout(0.2)(x)

    # Output layer with a single unit and sigmoid activation for binary classification
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()

    return model


def train_discriminator(
    output_units=OUTPUT_UNITS,
    num_units=NUM_UNITS,
    loss=LOSS,
    learning_rate=LEARNING_RATE,
):

    # generate training sequence
    inputs_real, targets_real = generate_training_sequences(SEQUENCE_LENGTH)
    inputs_fake, targets_fake = generate_training_sequences(
        SEQUENCE_LENGTH
    )  # Assuming fake data generation function exists

    # Combine real and fake data
    inputs = np.concatenate([inputs_real, inputs_fake])
    targets = np.concatenate([targets_real, targets_fake])

    # build the discriminator network
    model = build_discriminator(output_units, num_units, loss, learning_rate)

    # train the discriminator model
    history = model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train_discriminator()
