import keras
from pre_process import generate_training_sequences, SEQUENCE_LENGTH
import matplotlib.pyplot as plt

OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256, 256]
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model_bi_lstm.keras"
VALIDATION_SPLIT = 0.2


def build_model(output_units, num_units, loss, learning_rate):
    input = keras.layers.Input(shape=(None, output_units))

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(num_units[0], return_sequences=True)
    )(input)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(num_units[1]))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    model = build_model(output_units, num_units, loss, learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            SAVE_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        inputs,
        targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )

    _plot_history(history)


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(loc="lower right")

    axes[1].plot(history.history["loss"], label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Model Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("bi_lstm_training.png")
    plt.show()


if __name__ == "__main__":
    train()
