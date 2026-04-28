import json
import keras
import matplotlib.pyplot as plt
from pre_process import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 5
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"
HISTORY_PATH = "training_history.json"
PLOT_PATH = "training_progress.png"


def build_model(output_units, num_units, loss, learning_rate):
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"Training history saved to {HISTORY_PATH}")


def plot_history(history):
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LSTM Training Progress", fontsize=14)

    ax1.plot(epochs, history.history["loss"], "b-o", label="Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history.history["accuracy"], "g-o", label="Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"Training plot saved to {PLOT_PATH}")


def train(
    output_units=OUTPUT_UNITS,
    num_units=NUM_UNITS,
    loss=LOSS,
    learning_rate=LEARNING_RATE,
):
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    model = build_model(output_units, num_units, loss, learning_rate)

    history = model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save(SAVE_MODEL_PATH)
    save_history(history)
    plot_history(history)


if __name__ == "__main__":
    train()
