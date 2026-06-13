import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_history(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    return history


def plot_train_val_rmse(history, K1=3000, K2=12000, save_path=None):
    epoch = np.asarray(history["epoch"], dtype=float)

    train_rmse = np.asarray(history["train_rmse_total"], dtype=float)
    val_rmse = np.asarray(history["val_rmse_total"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(epoch, train_rmse, label="Training RMSE (17 simulator data points)",color="orange")
    ax.plot(epoch, val_rmse, label="Validation RMSE (10000 grid points)",color="blue")

    # Log-scale y-axis
    ax.set_yscale("log")

    # Stage boundaries
    if K1 is not None:
        ax.axvline(
            K1,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.7,
        )

    if K1 is not None and K2 is not None:
        ax.axvline(
            K1 + K2,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.7,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (kg)")
    ax.grid(True, axis="y", which="major", alpha=0.30)
    ax.grid(False, axis="x")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved figure at: {save_path}")

    plt.show()


if __name__ == "__main__":
    well="P1"
    history_path = Path(
        f"/home/lucas-kuhnen-costa/PycharmProjects/IOGR_StoAmaro/well_models/{well}/pinn_training_history.pkl"
    )

    save_path = Path(
        f"/home/lucas-kuhnen-costa/PycharmProjects/IOGR_StoAmaro/application/dissertation_figures/chapter4/{well}_train_val_rmse.pdf"
    )

    history = load_history(history_path)

    print("Available history keys:")
    for key in history.keys():
        print(f"  {key}")

    plot_train_val_rmse(
        history,
        save_path=save_path,
    )