import os

import numpy as np
import pandas as pd

from src.utils import Config


def load_and_process_files(directory, sensor_names, window_size):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith("_x.npy"):
            subject_id, scenario = filename.split("_")[:2]
            input_data, labels = load_data_files(directory, filename, subject_id, scenario)
            if input_data is not None and labels is not None:
                df = create_dataframe(input_data, labels, subject_id, scenario, sensor_names, window_size)
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def load_data_files(directory, filename, subject_id, scenario):
    try:
        input_data = np.load(os.path.join(directory, filename))
        labels = np.load(os.path.join(directory, f"{subject_id}_{scenario}_y.npy"))
        return input_data, labels
    except Exception as e:
        print(f"Error loading data for {filename}: {e}")
        return None, None


def create_dataframe(input_data, labels, subject_id, scenario, sensor_names, window_size):
    reshaped_input_data = input_data.reshape(input_data.shape[0], window_size * len(sensor_names))
    column_names = [f"{sensor}_{point}" for sensor in sensor_names for point in range(window_size)]
    df = pd.DataFrame(reshaped_input_data, columns=column_names)
    df.insert(0, "subject_id", subject_id)
    df.insert(1, "scenario", scenario)
    df.insert(2, "label", labels)
    return df


def generate_dataframes():
    print("Generating dataframes...")
    config = Config()

    output_dir = os.path.dirname(config.train_parquet_file)
    os.makedirs(output_dir, exist_ok=True)

    train_directory = os.path.join(output_dir, "preprocessed/train/motion/")
    test_directory = os.path.join(output_dir, "preprocessed/test/motion/")

    train_df = load_and_process_files(train_directory, config.sensor_names, config.window_size)
    test_df = load_and_process_files(
        test_directory,
        config.sensor_names,
        config.window_size,
    )

    train_df.to_parquet(config.train_parquet_file)
    test_df.to_parquet(config.test_parquet_file)

    print(f"Dataframes saved to {output_dir}")


# if __name__ == "__main__":
# generate_dataframes()
