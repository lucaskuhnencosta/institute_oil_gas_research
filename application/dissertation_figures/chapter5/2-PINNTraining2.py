import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from settings import *
def load_history(history_path):
    with open(history_path, "rb") as f:
        return pickle.load(f)


def collect_histories(project_root, wells):
    histories = {}

    for well in wells:
        history_path = (
            project_root
            / "well_models"
            / well
            / "pinn_training_history.pkl"
        )

        histories[well] = load_history(history_path)

    return histories


def stack_metric(histories, key):
    """
    Stack one metric from several well histories.

    Returns
    -------
    epoch : array, shape (n_min,)
    values : array, shape (n_wells, n_min)
    """

    min_len = min(len(h[key]) for h in histories.values())

    epoch = np.asarray(
        list(histories.values())[0]["epoch"][:min_len],
        dtype=float,
    )

    values = []

    for well, history in histories.items():
        values.append(
            np.asarray(history[key][:min_len], dtype=float)
        )

    values = np.vstack(values)

    return epoch, values

def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)

    if window <= 1:
        return x

    y = np.empty_like(x)

    for i in range(len(x)):
        i0 = max(0, i - window // 2)
        i1 = min(len(x), i + window // 2 + 1)
        y[i] = np.nanmean(x[i0:i1])

    return y


def smooth_rows(values, window=50):
    return np.vstack([
        moving_average(row, window=window)
        for row in values
    ])
def plot_mean_std_rmse(
    histories,
    K1=10000,
    K2=10000,
    save_path=None,
):
    # ------------------------------------------------------------
    # Collect train/validation RMSE
    # ------------------------------------------------------------
    epoch, train_values = stack_metric(histories, "train_rmse_total")
    _, val_values = stack_metric(histories, "val_rmse_total")

    # Remove nonpositive values for log-scale safety
    train_values = np.where(train_values > 0.0, train_values, np.nan)
    val_values = np.where(val_values > 0.0, val_values, np.nan)

    # Smooth each well first
    train_smooth = smooth_rows(train_values, window=100)
    val_smooth = smooth_rows(val_values, window=100)

    # Then compute mean and envelope
    train_mean = np.nanmean(train_smooth, axis=0)
    train_min = np.nanmin(train_smooth, axis=0)
    train_max = np.nanmax(train_smooth, axis=0)

    val_mean = np.nanmean(val_smooth, axis=0)
    val_min = np.nanmin(val_smooth, axis=0)
    val_max = np.nanmax(val_smooth, axis=0)
    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(
        epoch,
        train_min,
        train_max,
        color="blue",
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        epoch,
        val_min,
        val_max,
        color="orange",
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        epoch,
        train_mean,
        label="Training RMSE",
        color="blue",
        linewidth=1.8,
        zorder=3,
    )

    ax.plot(
        epoch,
        val_mean,
        label="Validation RMSE",
        color="orange",
        linewidth=1.8,
        zorder=4,
    )

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

    ax.set_xlabel("Epoch",fontsize=12)
    ax.set_ylabel("RMSE (kg)",fontsize=12)


    ax.grid(True, axis="y", which="major", alpha=0.30)
    ax.grid(False, axis="x")
    ax.legend(loc="lower left",frameon=False)
    ax.set_ylim(1e-0,1e3)
    ax.set_xlim(-50,23000)

    # ------------------------------------------------------------
    # Stage arrows
    # ------------------------------------------------------------
    y_arrow = 5e2  # adjust depending on your y-limits
    y_arrow2=1e2

    ax.annotate(
        "",
        xy=(K1, y_arrow),
        xytext=(0, y_arrow),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            linewidth=1.0,
        ),
        annotation_clip=False,
    )

    ax.annotate(
        "",
        xy=(K1+K2, y_arrow2),
        xytext=(0, y_arrow2),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            linewidth=1.0,
        ),
        annotation_clip=False,
    )

    ax.text(
        (K1+K2+3000) / 2,
        y_arrow2 * 1.10,
        "Adam",
        ha="center",
        va="bottom",
        fontsize=11,
    )


    ax.annotate(
        "",
        xy=(K1+K2+3000, y_arrow2),
        xytext=(K1+K2, y_arrow2),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            linewidth=1.0,
        ),
        annotation_clip=False,
    )

    ax.text(
        (K1+K2+1500),
        y_arrow2 * 1.10,
        "LGBFS",
        ha="center",
        va="bottom",
        fontsize=11,
    )


    ax.text(
        K1 / 2,
        y_arrow * 1.20,
        "Data only",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    ax.annotate(
        "",
        xy=(K1 + K2, y_arrow),
        xytext=(K1, y_arrow),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            linewidth=1.0,
        ),
        annotation_clip=False,
    )

    ax.text(
        K1 + K2 / 2,
        y_arrow * 1.20,
        "Data and physics",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure at: {save_path}")

    plt.show()


if __name__ == "__main__":

    project_root = Path(
        "/home/lucas-kuhnen-costa/PycharmProjects/IOGR_StoAmaro"
    )

    wells = ["P1","P2","P3","P4","P5","P6"]

    save_path = (
        project_root
        / "application"
        / "dissertation_figures"
        / "chapter5"
        / "PINN_training_validation.pdf"
    )

    histories = collect_histories(project_root, wells)

    plot_mean_std_rmse(
        histories,
        K1=10000,
        K2=10000,
        save_path=save_path,
    )