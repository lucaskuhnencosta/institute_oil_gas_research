import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from networks.networks import PINN
from utilities.block_builders import rebuild_pinn_from_weights
from configuration.wells import get_wells
from settings import *


@torch.no_grad()
def sweep_pinn(
    model: PINN,
    u1_min: float,
    u1_max: float,
    u2_min: float,
    u2_max: float,
    n_u1: int,
    n_u2: int,
    device: str = "cpu",
    batch_size: int = 8192,
):
    u1 = np.linspace(u1_min, u1_max, n_u1, dtype=np.float32)
    u2 = np.linspace(u2_min, u2_max, n_u2, dtype=np.float32)

    U1, U2 = np.meshgrid(u1, u2, indexing="ij")

    U_flat = np.stack(
        [U1.reshape(-1), U2.reshape(-1)],
        axis=1,
    ).astype(np.float32)

    U_t = torch.from_numpy(U_flat).to(device)

    ys = []
    for k in range(0, U_t.shape[0], batch_size):
        yb = model(U_t[k:k + batch_size])
        ys.append(yb.detach().cpu())

    Y_flat = torch.cat(ys, dim=0).numpy()
    Y = Y_flat.reshape(n_u1, n_u2, -1)

    return U1, U2, Y, U_flat, Y_flat


def load_sweep_ground_truth(sweep_path, output_names):
    with open(sweep_path, "rb") as f:
        sweep = pickle.load(f)

    u1_grid = np.asarray(sweep["u1_grid"], dtype=np.float32)
    u2_grid = np.asarray(sweep["u2_grid"], dtype=np.float32)

    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")

    Y_true = np.stack(
        [np.asarray(sweep["OUT"][name], dtype=np.float32) for name in output_names],
        axis=-1,
    )

    return U1, U2, Y_true


def plot_pinn_vs_reference_contours(
    U1,
    U2,
    Y_true,
    Y_pinn,
    output_names,
    save_path=None,
):
    error = Y_pinn - Y_true

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(9.0, 8.2),
        constrained_layout=True,
    )

    col_titles = ["Reference", "PINN", "Error"]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    for i, name in enumerate(output_names):
        Z_true = Y_true[:, :, i]
        Z_pinn = Y_pinn[:, :, i]
        Z_err = error[:, :, i]

        # Same levels for reference and PINN
        z_min = np.nanmin([np.nanmin(Z_true), np.nanmin(Z_pinn)])
        z_max = np.nanmax([np.nanmax(Z_true), np.nanmax(Z_pinn)])
        levels_state = np.linspace(z_min, z_max, 40)

        # Symmetric levels for error
        err_abs = np.nanpercentile(np.abs(Z_err), 99.0)
        err_abs = max(err_abs, 1e-12)
        levels_err = np.linspace(-err_abs, err_abs, 40)

        # Reference
        cf0 = axes[i, 0].contourf(
            U1,
            U2,
            Z_true,
            levels=levels_state,
            extend="both",
        )
        fig.colorbar(cf0, ax=axes[i, 0])

        # PINN
        cf1 = axes[i, 1].contourf(
            U1,
            U2,
            Z_pinn,
            levels=levels_state,
            extend="both",
        )
        fig.colorbar(cf1, ax=axes[i, 1])

        # Error
        cf2 = axes[i, 2].contourf(
            U1,
            U2,
            np.clip(Z_err, -err_abs, err_abs),
            levels=levels_err,
            extend="both",
        )
        fig.colorbar(cf2, ax=axes[i, 2])

        axes[i, 0].set_ylabel(name, fontsize=11)

        for j in range(3):
            ax = axes[i, j]
            ax.set_xlabel(r"$u_1$")
            ax.set_ylabel(r"$u_2$" if j == 0 else "")
            ax.set_box_aspect(1)
            ax.tick_params(axis="both", labelsize=8)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved figure at: {save_path}")

    plt.show()


if __name__ == "__main__":

    well_name = "P6"
    output_names = ["m_G_an", "m_G_t", "m_o_t"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wells = get_wells()
    well_dict = wells[well_name]

    y_min = well_dict["y_min"]
    y_max = well_dict["y_max"]

    project_root = Path(
        "/home/lucas-kuhnen-costa/PycharmProjects/IOGR_StoAmaro"
    )

    model_path = project_root / "well_models" / well_name / "PINN" / f"{well_name}.pth"

    sweep_path = (
        project_root
        / "well_models"
        / well_name
        / "sweep_results_figures_training.pkl"
    )

    save_path = (
        project_root
        / "application"
        / "dissertation_figures"
        / "chapter4"
        / f"{well_name}_pinn_vs_reference_states.pdf"
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = rebuild_pinn_from_weights(
        model_path,
        y_min=y_min,
        y_max=y_max,
        device=device,
    )
    model.eval()

    # ------------------------------------------------------------
    # Load reference sweep
    # ------------------------------------------------------------
    U1_ref, U2_ref, Y_true = load_sweep_ground_truth(
        sweep_path,
        output_names,
    )

    n_u1 = len(np.unique(U1_ref[:, 0]))
    n_u2 = len(np.unique(U2_ref[0, :]))

    # ------------------------------------------------------------
    # Sweep PINN on same grid
    # ------------------------------------------------------------
    U1_pinn, U2_pinn, Y_pinn, _, _ = sweep_pinn(
        model,
        u1_min=float(np.min(U1_ref)),
        u1_max=float(np.max(U1_ref)),
        u2_min=float(np.min(U2_ref)),
        u2_max=float(np.max(U2_ref)),
        n_u1=n_u1,
        n_u2=n_u2,
        device=device,
        batch_size=16384,
    )

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------
    print(f"Y_true shape: {Y_true.shape}")
    print(f"Y_pinn shape: {Y_pinn.shape}")

    rmse = np.sqrt(np.nanmean((Y_pinn - Y_true) ** 2, axis=(0, 1)))
    max_abs = np.nanmax(np.abs(Y_pinn - Y_true), axis=(0, 1))

    print("RMSE by state:")
    for name, value in zip(output_names, rmse):
        print(f"  {name}: {value:.6f}")

    print("MaxAbs by state:")
    for name, value in zip(output_names, max_abs):
        print(f"  {name}: {value:.6f}")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plot_pinn_vs_reference_contours(
        U1=U1_ref,
        U2=U2_ref,
        Y_true=Y_true,
        Y_pinn=Y_pinn,
        output_names=output_names,
        save_path=save_path,
    )