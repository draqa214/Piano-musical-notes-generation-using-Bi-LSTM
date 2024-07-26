# import keras
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv("synthetic_data.csv")

crisis_column = data["Crisis"]
y = data["Crisis"]

data = data.drop(columns=["Crisis"])

le = LabelEncoder()

for column in data.columns:
    if column != "Crisis":
        data[column] = le.fit_transform(data[column])
print(data.head())
X = data

print(X.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

print(X_train.shape)
model = Sequential()

model.add(
    LSTM(
        64, input_shape=(X_train.shape[1], 1), activation="relu", return_sequences=False
    )
)

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()
"""
input_layer = Input(shape=(X_train.shape[1], 1))
transformer_layer = tf.keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16, value_dim=16
)(input_layer, input_layer)
transformer_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_layer)
transformer_output = Dense(32, activation="relu")(transformer_output)
transformer_output = Dense(1, activation="sigmoid")(transformer_output)

transformer_model = keras.models.Model(inputs=input_layer, outputs=transformer_output)

transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

"""
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

history = model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.2)
# transformer_history = transformer_model.fit(
#    X_train, y_train, epochs=10, batch_size=1, validation_split=0.2
# )

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
# plt.plot(transformer_history.history["loss"], label="Transformer Training Loss")
# plt.plot(transformer_history.history["val_loss"], label="Transformer Validation Loss")
plt.title("Transformer Model Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


accuracy = model.evaluate(X_test, y_test)[1]
# transformer_accuracy = transformer_model.evaluate(X_test, y_test)[1]

print(f"Test Accuracy: {accuracy * 100:.2f}%")
# print(f"Transformer Test Accuracy: {transformer_accuracy * 100:.2f}%")
