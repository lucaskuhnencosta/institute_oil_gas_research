import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from settings import *
def load_history(history_path):
    with open(history_path, "rb") as f:
        return pickle.load(f)

def stack_grouped_rmse(histories, prefix, group):
    """
    Reconstruct grouped RMSE from saved component RMSEs.

    prefix:
        "train" or "val"

    group:
        "pressure" -> z0,z1 = P_bh_bar, P_tb_b_bar
        "flow"     -> z2,z3 = w_G_inj, w_res
    """

    if group == "pressure":
        keys = [f"{prefix}_rmse_z0", f"{prefix}_rmse_z1"]
    elif group == "flow":
        keys = [f"{prefix}_rmse_z2", f"{prefix}_rmse_z3"]
    else:
        raise ValueError(f"Unknown group: {group}")

    min_len = min(
        min(len(h[k]) for k in keys)
        for h in histories.values()
    )

    epoch = np.asarray(
        list(histories.values())[0]["epoch"][:min_len],
        dtype=float,
    )

    values = []

    for well, history in histories.items():
        comps = np.vstack([
            np.asarray(history[k][:min_len], dtype=float)
            for k in keys
        ])

        grouped = np.sqrt(np.nanmean(comps ** 2, axis=0))
        values.append(grouped)

    values = np.vstack(values)

    return epoch, values

def collect_histories(project_root, wells):
    histories = {}

    for well in wells:
        history_path = (
            project_root
            / "well_models"
            / well
            / "algnn_training_history.pkl"
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
def plot_algnn_grouped_rmse_single_plot(
    histories,
    K1=10000,
    K2=10000,
    save_path=None,
):
    # ------------------------------------------------------------
    # Reconstruct grouped RMSEs from saved component RMSEs
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Pressure RMSE: z0=P_bh_bar, z1=P_tb_b_bar
    # ------------------------------------------------------------
    epoch, train_z0 = stack_metric(histories, "train_rmse_z0")
    _, train_z1 = stack_metric(histories, "train_rmse_z1")

    _, val_z0 = stack_metric(histories, "val_rmse_z0")
    _, val_z1 = stack_metric(histories, "val_rmse_z1")

    train_pressure = np.sqrt((train_z0 ** 2 + train_z1 ** 2) / 2.0)
    val_pressure = np.sqrt((val_z0 ** 2 + val_z1 ** 2) / 2.0)

    # ------------------------------------------------------------
    # Flow RMSE: z2=w_G_inj, z3=w_res
    # ------------------------------------------------------------
    _, train_z2 = stack_metric(histories, "train_rmse_z2")
    _, train_z3 = stack_metric(histories, "train_rmse_z3")

    _, val_z2 = stack_metric(histories, "val_rmse_z2")
    _, val_z3 = stack_metric(histories, "val_rmse_z3")

    train_flow = np.sqrt((train_z2 ** 2 + train_z3 ** 2) / 2.0)
    val_flow = np.sqrt((val_z2 ** 2 + val_z3 ** 2) / 2.0)

    train_pressure = np.where(train_pressure > 0.0, train_pressure, np.nan)
    val_pressure = np.where(val_pressure > 0.0, val_pressure, np.nan)
    train_flow = np.where(train_flow > 0.0, train_flow, np.nan)
    val_flow = np.where(val_flow > 0.0, val_flow, np.nan)

    train_pressure_s = smooth_rows(train_pressure, window=50)
    val_pressure_s = smooth_rows(val_pressure, window=50)
    train_flow_s = smooth_rows(train_flow, window=50)
    val_flow_s = smooth_rows(val_flow, window=50)

    def mean_min_max(values):
        return (
            np.nanmean(values, axis=0),
            np.nanmin(values, axis=0),
            np.nanmax(values, axis=0),
        )

    train_pressure_mean, train_pressure_min, train_pressure_max = mean_min_max(train_pressure_s)
    val_pressure_mean, val_pressure_min, val_pressure_max = mean_min_max(val_pressure_s)

    train_flow_mean, train_flow_min, train_flow_max = mean_min_max(train_flow_s)
    val_flow_mean, val_flow_min, val_flow_max = mean_min_max(val_flow_s)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # Pressure bands
    ax.fill_between(epoch, train_pressure_min, train_pressure_max,
                    color="blue", alpha=0.10, linewidth=0, zorder=1)

    ax.fill_between(epoch, val_pressure_min, val_pressure_max,
                    color="orange", alpha=0.12, linewidth=0, zorder=1)

    # Flow bands
    ax.fill_between(epoch, train_flow_min, train_flow_max,
                    color="blue", alpha=0.06, linewidth=0, zorder=1)

    ax.fill_between(epoch, val_flow_min, val_flow_max,
                    color="orange", alpha=0.08, linewidth=0, zorder=1)

    # Pressure mean curves: solid
    p_train, = ax.plot(
        epoch, train_pressure_mean,
        label="Training pressure RMSE (bar)",
        color="blue", linestyle="-", linewidth=1.8, zorder=4
    )

    p_val, = ax.plot(
        epoch, val_pressure_mean,
        label="Validation pressure RMSE (bar)",
        color="orange", linestyle="-", linewidth=1.8, zorder=5
    )

    # Flow mean curves: dashed
    w_train, = ax.plot(
        epoch, train_flow_mean,
        label="Training flow RMSE (kg/s)",
        color="blue", linestyle="--", linewidth=1.8, zorder=6
    )

    w_val, = ax.plot(
        epoch, val_flow_mean,
        label="Validation flow RMSE (kg/s)",
        color="orange", linestyle="--", linewidth=1.8, zorder=7
    )


    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------
    ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)

    ax.set_xlim(-50, 20000)

    # Adjust after seeing the plot
    ax.set_ylim(3e-3, 10)

    ax.grid(True, axis="y", which="major", alpha=0.30)
    ax.grid(False, axis="x")

    # Stage boundary
    if K1 is not None:
        ax.axvline(
            K1,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.7,
        )

    # ------------------------------------------------------------
    # Stage arrows
    # ------------------------------------------------------------
    y_arrow = 1.0

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

    ax.text(
        K1 / 2,
        y_arrow * 1.20,
        "Adam",
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
        "LBFGS",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    # Combined legend
    lines = [p_train, p_val, w_train, w_val]
    labels = [line.get_label() for line in lines]

    ax.legend(
        lines,
        labels,
        loc="lower left",
        frameon=False,
        fontsize=9,
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
        / "AlgNN_training_validation.pdf"
    )

    histories = collect_histories(project_root, wells)

    plot_algnn_grouped_rmse_single_plot(
        histories,
        K1=10000,
        K2=10000,
        save_path=save_path,
    )