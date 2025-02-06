from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(parquet_file: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_file)


def z_normalize_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    for _, columns in sensor_columns.items():
        train_df.loc[:, columns] = scaler.fit_transform(train_df.loc[:, columns])
        val_df.loc[:, columns] = scaler.transform(val_df.loc[:, columns])
        test_df.loc[:, columns] = scaler.transform(test_df.loc[:, columns])
    return train_df, val_df, test_df


def create_ds(df: pd.DataFrame, sensor_columns: Dict[str, List[str]]) -> tf.data.Dataset:
    data_ds = prepare_sensor_data(df, sensor_columns)
    label_dataset = tf.data.Dataset.from_tensor_slices(df["label"].values)
    return tf.data.Dataset.zip((data_ds, label_dataset))


def prepare_sensor_data(df: pd.DataFrame, sensor_columns: Dict[str, List[str]]) -> tf.data.Dataset:
    combined_sensor_data = np.concatenate([df[cols].values for cols in sensor_columns.values()], axis=1)
    num_data_points, num_sensors = combined_sensor_data.shape[0], len(sensor_columns)
    window_size = combined_sensor_data.shape[1] // num_sensors
    reshaped_data = combined_sensor_data.reshape((num_data_points, window_size, num_sensors))
    return tf.data.Dataset.from_tensor_slices(reshaped_data)


def split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_subject_id: str,
    sensor_columns: Dict[str, List[str]],
    validation_size: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    test_df = test_df[test_df["subject_id"] == test_subject_id]
    remaining_df = train_df[train_df["subject_id"] != test_subject_id]

    unique_subjects = remaining_df["subject_id"].unique()
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=validation_size, random_state=42)

    assert len(train_subjects) == 11
    assert len(val_subjects) == 3

    #! Split by subject_id
    train_df = remaining_df[remaining_df["subject_id"].isin(train_subjects)]
    val_df = remaining_df[remaining_df["subject_id"].isin(val_subjects)]

    train_df, val_df, test_df = z_normalize_data(train_df, val_df, test_df, sensor_columns)

    return (
        create_ds(train_df, sensor_columns),
        create_ds(val_df, sensor_columns),
        create_ds(test_df, sensor_columns),
    )
