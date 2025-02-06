import argparse
import logging
import os
import time
import warnings
from typing import Tuple

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tqdm import tqdm  # noqa: E402

# local imports
from src.preprocessing.generate_dataframes import generate_dataframes  # noqa: E402
from src.utils import (  # noqa: E402
    configure_for_performance,
    create_run_directory,
    evaluate_model,
    get_callbacks,
    get_model,
    get_optimizer,
    load_data,
    log_metrics,
    plot_confusion_matrix,
    save_config,
    save_initial_model,
    save_metrics,
    save_model,
    save_model_comparison_file,
    split_data,
    visualize_losses,
    visualize_results,
)
from src.utils.config import Config  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Enable memory growth

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Config,
    subject_dir: str,
) -> Tuple[int, dict]:
    callbacks = get_callbacks(subject_dir)

    print("\n" * 3, "-" * 50, "Model training starts now", "-" * 50)
    start_time = time.time()
    history = model.fit(
        # shuffle buffer is the size of the length of the train dataset
        configure_for_performance(config, train_ds, is_training=True),
        validation_data=configure_for_performance(config, val_ds, is_training=False),
        epochs=config.epochs,
        verbose=config.verbose,
        callbacks=callbacks,
    )
    training_time = int(time.time() - start_time)

    losses = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }

    return training_time, losses


def train_and_evaluate(config: Config):
    train_df = load_data(config.train_parquet_file)
    test_df = load_data(config.test_parquet_file)

    sensor_columns = {
        sensor: [col for col in train_df.columns if col.startswith(sensor)] for sensor in config.sensor_names
    }
    unique_subject_ids = train_df["subject_id"].unique()

    if config.train_one_subject_only:
        unique_subject_ids = unique_subject_ids[:1]

    with tqdm(total=len(config.models_to_train), desc="Model Training", unit="model") as model_pbar:
        for model_type in config.models_to_train:
            run_dir, timestamp = create_run_directory(model_type, config.base_output_dir)
            save_config(config, run_dir)

            metrics_subjects_df = pd.DataFrame(
                columns=[
                    "subject_id",
                    "precision",
                    "recall",
                    "accuracy",
                    "f1_score",
                ]
            )
            total_training_time = 0
            subjects_losses = {"train_loss": [], "val_loss": []}

            # optimizer is reloaded each subject, this is on purpose. Retrain each iteration with same weights -> ignore the tf error
            optimizer = get_optimizer(config.optimizer)(
                learning_rate=config.learning_rate, weight_decay=config.weight_decay, clipnorm=config.clipnorm
            )
            input_shape = (config.window_size, len(sensor_columns))
            model = get_model(input_shape, model_type)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"],
            )
            initial_model_path = save_initial_model(model, run_dir)

            if config.print_model_summary:
                model.summary()

            with tqdm(
                total=len(unique_subject_ids),
                desc=f"Training {model_type} Model",
                unit="subject",
            ) as pbar:
                for test_subject_id in unique_subject_ids:
                    subject_dir = os.path.join(run_dir, f"subject_{test_subject_id}")
                    os.makedirs(subject_dir, exist_ok=True)

                    train_ds, val_ds, test_ds = split_data(
                        train_df=train_df,
                        test_df=test_df,
                        test_subject_id=test_subject_id,
                        sensor_columns=sensor_columns,
                        validation_size=config.validation_size,
                    )

                    if config.train_subset:  #! For testing purposes, only use 3 batches of the train data
                        train_ds = train_ds.take(3)

                    try:
                        model = tf.keras.models.load_model(initial_model_path)
                    except Exception as e:
                        logging.error(f"Could not load best weights for subject {test_subject_id}: {e}")
                        print(e.with_traceback(None))
                        break

                    training_time_subject, subject_loss = train_model(model, train_ds, val_ds, config, subject_dir)
                    total_training_time += training_time_subject

                    try:
                        model.load_weights(os.path.join(subject_dir, "best.weights.h5"))
                    except Exception as e:
                        logging.error(f"Could not load best weights for subject {test_subject_id}: {e}")
                        break

                    save_model(model, run_dir, test_subject_id)

                    subjects_losses["train_loss"].append(subject_loss["train_loss"])
                    subjects_losses["val_loss"].append(subject_loss["val_loss"])

                    metrics_subject = evaluate_model(config, model, test_ds, test_subject_id)
                    metrics_subject["val_loss"] = min(subject_loss["val_loss"])
                    metrics_subjects_df = log_metrics(metrics_subject, metrics_subjects_df)

                    plot_confusion_matrix(metrics_subject["conf_matrix"], test_subject_id, run_dir)

                    pbar.update(1)

            save_metrics(metrics_subjects_df, run_dir)
            visualize_results(metrics_subjects_df, run_dir)
            visualize_losses(subjects_losses, unique_subject_ids, run_dir)

            save_model_comparison_file(
                model_type,
                total_training_time,
                timestamp,
                config.train_one_subject_only,
                config.train_subset,
                config.epochs,
                config.learning_rate,
                config.weight_decay,
                metrics_subjects_df["accuracy"].mean(),
                metrics_subjects_df["f1_score"].mean(),
                config.base_output_dir,
            )

            model_pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Training script with optional learning rate and weight decay")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--wd", type=float, help="Weight decay")
    parser.add_argument("--train_one_subject_only", action="store_true", help="Train one subject only")
    parser.add_argument("--train_subset", action="store_true", help="Use only 3 batches per subject")
    args = parser.parse_args()

    config = Config()

    if args.train_one_subject_only:
        config.train_one_subject_only = True
    if args.train_subset:
        config.train_subset = True

    if not os.path.exists(config.train_parquet_file) or not os.path.exists(config.test_parquet_file):
        try:
            generate_dataframes()
        except Exception as e:
            print(f"Error loading and processing files: {e}")
            print("----------------------------------------------------")
            print("Download the dataset as dataframes from the prerequisites in the README")
            print("----------------------------------------------------")
            return

    if args.lr is not None:
        config.learning_rate = args.lr
    if args.wd is not None:
        config.weight_decay = args.wd

    print(f"\nTraining with lr={config.learning_rate:.2e}, wd={config.weight_decay:.2e}")
    train_and_evaluate(config)

    model_comparison_path = os.path.join(config.base_output_dir, "model_comparison.tsv")
    print(pd.read_csv(model_comparison_path, sep="\t"))


if __name__ == "__main__":
    main()
