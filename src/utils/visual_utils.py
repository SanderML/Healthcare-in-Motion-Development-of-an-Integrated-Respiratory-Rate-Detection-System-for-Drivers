import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_results(metrics_df: pd.DataFrame, run_dir: str):
    metrics_df.plot(x="subject_id", y="f1_score", kind="bar", figsize=(10, 6))
    plt.title("Average Arithmetic Mean by Subject")
    plt.xlabel("Subject ID")
    plt.ylabel("Average Arithmetic Mean")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "performance_metrics.png"))
    plt.close()


def pad_losses(losses: list, pad_value: float = np.nan) -> np.ndarray:
    max_length = max(len(loss) for loss in losses)
    padded_losses = [loss + [pad_value] * (max_length - len(loss)) for loss in losses]
    return np.array(padded_losses)


def visualize_losses(all_losses: dict, subject_ids: list, run_dir: str):
    # Save individual subject losses
    for i, subject_id in enumerate(subject_ids):
        plt.figure(figsize=(12, 6))
        plt.plot(all_losses["train_loss"][i], label="Train Loss")
        plt.plot(all_losses["val_loss"][i], label="Val Loss")

        if max(all_losses["train_loss"][i]) > 1:
            plt.ylim(0, 1)
        plt.title(f"Subject {subject_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        subject_dir = os.path.join(run_dir, f"subject_{subject_id}")
        os.makedirs(subject_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, "losses.png"))
        plt.close()

    # Pad losses and calculate averages
    train_losses_padded = pad_losses(all_losses["train_loss"])
    val_losses_padded = pad_losses(all_losses["val_loss"])

    avg_train_loss = np.nanmean(train_losses_padded, axis=0)
    avg_val_loss = np.nanmean(val_losses_padded, axis=0)

    # Save average losses plot
    plt.figure(figsize=(12, 6))
    plt.plot(avg_train_loss, label="Avg Train Loss")
    plt.plot(avg_val_loss, label="Avg Val Loss")

    if max(avg_train_loss) > 1:
        plt.ylim(0, 1)
    plt.title("Average Loss Across All Subjects")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "average_losses.png"))
    plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray, subject_id: str, run_dir: str):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for Subject {subject_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    subject_dir = os.path.join(run_dir, f"subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)
    plt.savefig(os.path.join(subject_dir, "confusion_matrix.png"))
    plt.close()
