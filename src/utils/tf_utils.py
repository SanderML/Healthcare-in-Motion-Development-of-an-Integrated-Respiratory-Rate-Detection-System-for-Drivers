import os
from typing import Tuple

import tensorflow as tf
from keras.api.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.api.optimizers import Adadelta, Adafactor, Adam, Adamax, AdamW

from src.models import (
    ResNet,
    create_cnn_bilstm_model,
    create_tiny_test_model,
)


def get_model(input_shape: Tuple[int], model_type: str) -> tf.keras.Model:
    model_dict = {
        "tiny": create_tiny_test_model,
        "cnn_bilstm": create_cnn_bilstm_model,
    }

    if model_type == "resnet":
        return ResNet(input_shape, num_filters=16, dropout_rate=0.5).ResNet18()
    if model_type in model_dict:
        return model_dict[model_type](input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_optimizer(optimizer: str) -> tf.keras.optimizers.Optimizer:
    optimizer_dict = {
        "adam": Adam,
        "adadelta": Adadelta,
        "adamax": Adamax,
        "adamw": AdamW,
        "adafactor": Adafactor,
    }
    if optimizer in optimizer_dict:
        return optimizer_dict[optimizer]
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def configure_for_performance(config, ds, is_training):
    ds = ds.cache()
    if is_training:
        ds = ds.shuffle(config.max_train_ds)
    ds = ds.batch(config.batch_size if is_training else config.batch_size * 2)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


class EpochMessageCallback(Callback):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            model_metrics = {
                k: round(v, 2) for k, v in logs.items() if k in ["loss", "val_loss", "accuracy", "val_accuracy"]
            }
            print("-" * 5, f"Epoch {epoch + 1}", model_metrics, "-" * 5, "\n")


def get_callbacks(subject_dir):
    csv_logger = CSVLogger(os.path.join(subject_dir, "training_log.csv"))
    model_checkpoint = ModelCheckpoint(
        os.path.join(subject_dir, "best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    epoch_message_callback = EpochMessageCallback(interval=5)

    terminate_on_nan = TerminateOnNaN()
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, start_from_epoch=10, verbose=0, min_delta=0.001)

    callbacks = [
        csv_logger,
        model_checkpoint,
        epoch_message_callback,
        terminate_on_nan,
        early_stopping,
    ]

    return callbacks


def save_model(model: tf.keras.Model, run_dir: str, test_subject_id: str):
    subject_dir = os.path.join(run_dir, f"subject_{test_subject_id}")
    os.makedirs(subject_dir, exist_ok=True)
    model.save(os.path.join(subject_dir, "full_model.keras"))


def save_initial_model(model: tf.keras.Model, run_dir: str) -> str:
    initial_model_path = os.path.join(run_dir, "initial_model.keras")
    model.save(initial_model_path)
    tf.keras.utils.plot_model(model, to_file=os.path.join(run_dir, "model_architecture.png"), show_shapes=True)
    return initial_model_path
