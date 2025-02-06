from dataclasses import dataclass, field
from typing import List


def get_sensor_names() -> List[str]:
    return ["piezo", "accelo", "video_belt", "video_chest"]


def models_to_train():
    models = []

    # models = ["cnn_bilstm"]
    # models = ["resnet"]
    models = ["resnet", "cnn_bilstm"] * 1

    return models


@dataclass
class Config:
    # * ----- Until mentioned otherwise, do NOT change variables-----
    train_one_subject_only: bool = False  #! DO NOT TOUCH, change with flag --train_one_subject_only
    train_subset: bool = False  #! DO NOT TOUCH, change with flag --train_subset
    # * ----- Before this line, do NOT touch anything -----

    # * Parameters to tweak
    epochs: int = 50
    learning_rate: float = 5e-3  # * gets overwritten if running jobs
    weight_decay: float = 1e-4  # * gets overwritten if running jobs
    optimizer: str = "adamw"
    batch_size: int = 512
    clipnorm: float = 1.0

    ## Verbose
    verbose: int = 0
    print_model_summary: bool = False

    # Project "constants"
    train_parquet_file: str = "data/input_data_with_labels_motion_train.parquet"
    test_parquet_file: str = "data/input_data_with_labels_motion_test.parquet"
    base_output_dir: str = "out/"
    window_size: int = 201
    validation_size: float = 0.2  # split on subjects: train 11, val 3, test 1
    max_train_ds: int = 150166  # longest train_ds length

    sensor_names: List[str] = field(default_factory=get_sensor_names)
    models_to_train: List[str] = field(default_factory=models_to_train)
