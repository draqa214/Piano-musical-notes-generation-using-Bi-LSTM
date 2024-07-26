import keras
from pre_process import generate_training_sequences, SEQUENCE_LENGTH
import matplotlib.pyplot as plt

OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256, 256]  # Increase the number of units
EPOCHS = 5
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model_bi_lstm.h5"


def build_model(output_units, num_units, loss, learning_rate):
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

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()

    return model


def train(
    output_units=OUTPUT_UNITS,
    num_units=NUM_UNITS,
    loss=LOSS,
    learning_rate=LEARNING_RATE,
):
    # generate training sequence
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    history = model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history["accuracy"], label="Accuracy (training data)")
    # plt.plot(history.history["val_accuracy"], label="Accuracy (validation data)")
    # plt.title("Model Accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Epoch")
    # plt.legend(loc="lower right")
    # plt.savefig("accuracy_plot.png")
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history["loss"], label="Loss (training data)")
    # plt.plot(history.history["val_loss"], label="Loss (validation data)")
    # plt.title("Model Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(loc="upper right")
    # plt.savefig("loss_plot.png")
    # plt.show()
    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
