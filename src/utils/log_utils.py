import datetime
import json
import os
import types
from dataclasses import fields
from typing import Tuple

import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import Config
from src.utils.tf_utils import configure_for_performance


def create_run_directory(model_type: str, base_output_dir: str) -> Tuple[str, str]:
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    run_dir = os.path.join(base_output_dir, f"{timestamp}-{model_type}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def log_metrics(metrics: dict, all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = pd.DataFrame([metrics])
    val_loss_column = metrics_df.pop("val_loss")
    metrics_df.insert(1, "val_loss", val_loss_column)

    # Remove empty or all-NA columns from the metrics DataFrame before concatenation
    metrics_df = metrics_df.dropna(axis=1, how="all")

    if all_metrics_df.empty:
        return metrics_df
    return pd.concat([all_metrics_df, metrics_df], ignore_index=True)


def save_model_comparison_file(
    model_name: str,
    training_time: int,
    timestamp: str,
    train_one_subject_only: bool,
    train_subset: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    accuracy: float,
    f1_score: float,
    base_output_dir: str,
):
    date, time = timestamp.split("_")

    # Create readable timestamp for filtering
    timestamp_readable = f"20{date}-{time}"

    # Determine dataset mode
    if train_one_subject_only and train_subset:
        dataset_mode = "single_subject_subset"
    elif train_one_subject_only:
        dataset_mode = "single_subject"
    elif train_subset:
        dataset_mode = "full_subset"
    else:
        dataset_mode = "full_losocv"

    new_data = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "training_time": training_time,
                "dataset_mode": dataset_mode,
                "timestamp": timestamp_readable,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "f1_score": str(round(f1_score, 4)),
            }
        ],
    )

    comparison_file = os.path.join(base_output_dir, "model_comparison.tsv")
    if os.path.exists(comparison_file):
        existing_df = pd.read_csv(comparison_file, sep="\t")
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        combined_df = new_data

    combined_df.to_csv(comparison_file, index=False, sep="\t", float_format="%.1e")


def save_metrics(metrics_df: pd.DataFrame, run_dir: str):
    all_metrics_file = os.path.join(run_dir, "all_metrics.tsv")
    metrics_df.to_csv(all_metrics_file, index=False, sep="\t")


def save_config(config: Config, run_dir: str):
    config_path = os.path.join(run_dir, "config.json")
    # Convert dataclass to dict, excluding any non-serializable objects
    config_dict = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if not isinstance(getattr(config, field.name), (type, types.FunctionType))
    }
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_dict, config_file, indent=4)


def evaluate_model(config: Config, model: tf.keras.Model, test_ds: tf.data.Dataset, subject_id: str) -> dict:
    test_predictions = model.predict(configure_for_performance(config, test_ds, is_training=False), verbose=0).round()
    test_labels = [label.numpy() for _, label in test_ds]

    precision = round(precision_score(test_labels, test_predictions, zero_division=0), 4)
    recall = round(recall_score(test_labels, test_predictions, zero_division=0), 4)
    accuracy = round(accuracy_score(test_labels, test_predictions), 4)
    f1 = round(f1_score(test_labels, test_predictions, zero_division=0), 4)

    conf_matrix = confusion_matrix(test_labels, test_predictions)

    return {
        "subject_id": subject_id,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "conf_matrix": conf_matrix,
    }
