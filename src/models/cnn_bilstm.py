from typing import Tuple

import tensorflow as tf
from keras import Sequential
from keras.api.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    InputLayer,
    MaxPooling1D,
    SpatialDropout1D,
)


def create_tiny_test_model(input_shape: Tuple[int]) -> tf.keras.Model:
    """Creates a minimal CNN model for testing the training pipeline."""
    model = Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv1D(filters=8, kernel_size=3, padding="same"),
            MaxPooling1D(pool_size=2),
            GlobalAveragePooling1D(),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def create_cnn_bilstm_model(input_shape: Tuple[int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            InputLayer(input_shape=input_shape),
            #
            Conv1D(filters=32, kernel_size=7, padding="same", kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation("relu"),
            SpatialDropout1D(0.2),
            #
            Conv1D(filters=64, kernel_size=5, padding="same", kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation("relu"),
            SpatialDropout1D(0.2),
            #
            Bidirectional(LSTM(128, return_sequences=True, activation="tanh")),
            BatchNormalization(),
            Dropout(0.2),
            #
            Bidirectional(LSTM(64, return_sequences=True, activation="tanh")),
            BatchNormalization(),
            Dropout(0.2),
            #
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model
