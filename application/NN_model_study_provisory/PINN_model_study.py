from application.plotting_engine import plot_surface
from utilities.block_builders import *
from networks.networks import PINN
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from simulators.surrogate_simulator.surrogate_model_torch import glc_surrogate_dx_torch

@torch.no_grad()
def sweep_pinn(
    model: PINN,
    u1_min: float,
    u1_max: float,
    u2_min: float,
    u2_max: float,
    n_u1: int = 81,
    n_u2: int = 81,
    device: str = "cpu",
    batch_size: int = 8192,
):
    """
    Sweeps (u1,u2) over a grid and returns:
      U1, U2 meshgrids  -> shape (n_u2, n_u1)
      Y -> shape (n_u2, n_u1, 3)
    """
    u1 = np.linspace(u1_min, u1_max, n_u1, dtype=np.float32)
    u2 = np.linspace(u2_min, u2_max, n_u2, dtype=np.float32)
    U1, U2 = np.meshgrid(u1, u2, indexing="ij")

    # pts = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)  # (N,2)
    # pts_t = torch.from_numpy(pts).to(device)
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

@torch.no_grad()
def evaluate_physics_on_sweep(
    U_flat: np.ndarray,
    Y_flat: np.ndarray,
    well_dict: dict,
    device: str = "cpu",
    batch_size: int = 8192,
):
    """
    Evaluates dx = f(y_hat, u) on a swept PINN prediction.

    Returns
    -------
    dx_flat : array, shape (N, 3)
    """

    U_t = torch.from_numpy(U_flat.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y_flat.astype(np.float32)).to(device)

    BSW = well_dict["BSW"]
    GOR = well_dict["GOR"]
    PI = well_dict["PI"]
    K_gs = well_dict["K_gs"]
    K_inj = well_dict["K_inj"]
    K_pr = well_dict["K_pr"]

    dxs = []

    for k in range(0, U_t.shape[0], batch_size):
        u_b = U_t[k:k + batch_size]
        y_b = Y_t[k:k + batch_size]

        physics_out = glc_surrogate_dx_torch(
            y=y_b,
            u=u_b,
            BSW=BSW,
            GOR=GOR,
            PI=PI,
            K_gs=K_gs,
            K_inj=K_inj,
            K_pr=K_pr,
            return_z=False,
        )

        if isinstance(physics_out, tuple):
            dx_b = physics_out[0]
        else:
            dx_b = physics_out

        dxs.append(dx_b.detach().cpu())

    dx_flat = torch.cat(dxs, dim=0).numpy()

    return dx_flat

def build_bad_physics_mask(
    dx_flat: np.ndarray,
):
    """
    Builds masks using raw residual thresholds in kg/s.
    """

    abs_dx = np.abs(dx_flat)

    good_mask_flat = (
        (abs_dx[:, 0] <= 40)
        & (abs_dx[:, 1] <= 40)
        & (abs_dx[:, 2] <= 40)
    )

    bad_mask_flat = ~good_mask_flat

    return good_mask_flat, bad_mask_flat


def plot_physics_residual_maps(
    U1,
    U2,
    dx_flat,
    bad_mask_flat,
    n_u1,
    n_u2,
    save_path=None,
):
    """
    Plots dx1, dx2, dx3 and ||dx|| with bad-region contour.
    Uses percentile-based color limits to avoid huge outliers dominating the plot.
    """

    DX = dx_flat.reshape(n_u1, n_u2, 3)

    DX1 = DX[:, :, 0]
    DX2 = DX[:, :, 1]
    DX3 = DX[:, :, 2]
    DXN = np.linalg.norm(DX, axis=2)

    BAD = bad_mask_flat.reshape(n_u1, n_u2).astype(float)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.4),
        constrained_layout=True,
    )

    fields = [
        (DX1, r"$dx_1 = w_{G,in} - w_{G,inj}$", True),
        (DX2, r"$dx_2 = w_{G,inj} + w_{G,res} - w_{G,out}$", True),
        (DX3, r"$dx_3 = w_{L,res} - w_{L,out}$", True),
        (DXN, r"$||dx||_2$", False),
    ]

    for ax, (Z, title, signed_field) in zip(axes.ravel(), fields):

        # --------------------------------------------------
        # Robust color limits
        # --------------------------------------------------
        if signed_field:
            vmax = np.nanpercentile(np.abs(Z), 99.0)
            vmax = max(vmax, 1e-12)
            vmin = -vmax
            levels = np.linspace(vmin, vmax, 41)
        else:
            vmin = 0.0
            vmax = np.nanpercentile(Z, 99.0)
            vmax = max(vmax, 1e-12)
            levels = np.linspace(vmin, vmax, 41)

        # Optional: clip only for visualization
        Z_plot = np.clip(Z, vmin, vmax)

        cf = ax.contourf(
            U1,
            U2,
            Z_plot,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )

        # Bad-region boundary
        ax.contour(
            U1,
            U2,
            BAD,
            levels=[0.5],
            colors="k",
            linewidths=1.5,
        )

        fig.colorbar(cf, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$u_1$")
        ax.set_ylabel(r"$u_2$")
        ax.set_box_aspect(1)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved figure at: {save_path}")

    plt.show()


def print_worst_points(U_flat, Y_flat, dx_flat, top_k=20):
    """
    Prints the worst residual points.
    """

    dx_norm = np.linalg.norm(dx_flat, axis=1)
    idx_sorted = np.argsort(dx_norm)[::-1]

    print("=" * 100)
    print(f"Worst {top_k} physics residual points")
    print("=" * 100)

    for rank, idx in enumerate(idx_sorted[:top_k], start=1):
        print(
            f"{rank:02d} | idx={idx:05d} | "
            f"u={U_flat[idx]} | "
            f"y_pred={Y_flat[idx]} | "
            f"dx={dx_flat[idx]} | "
            f"||dx||={dx_norm[idx]:.6e}"
        )

if __name__ == "__main__":

    wells = get_wells()

    well_name = "P4"
    well_dict = wells[well_name]

    y_min = well_dict["y_min"]
    y_max = well_dict["y_max"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = f"../../well_models/{well_name}/PINN/{well_name}.pth"

    model = rebuild_pinn_from_weights(
        model_path,
        y_min=y_min,
        y_max=y_max,
        device=device,
    )

    model.eval()

    u1_min, u1_max = U1_MIN, U1_MAX
    u2_min, u2_max = U2_MIN, U2_MAX

    n_u1 = 101
    n_u2 = 101

    # ------------------------------------------------------------
    # 1. Sweep PINN
    # ------------------------------------------------------------
    U1, U2, Y, U_flat, Y_flat = sweep_pinn(
        model,
        u1_min=u1_min,
        u1_max=u1_max,
        u2_min=u2_min,
        u2_max=u2_max,
        n_u1=n_u1,
        n_u2=n_u2,
        device=device,
        batch_size=16384,
    )

    print(f"Y shape: {Y.shape}")
    print(f"y_min from PINN sweep: {np.min(Y_flat, axis=0)}")
    print(f"y_max from PINN sweep: {np.max(Y_flat, axis=0)}")

    # ------------------------------------------------------------
    # 2. Evaluate physics residuals
    # ------------------------------------------------------------
    dx_flat = evaluate_physics_on_sweep(
        U_flat=U_flat,
        Y_flat=Y_flat,
        well_dict=well_dict,
        device=device,
        batch_size=16384,
    )

    dx_norm = np.linalg.norm(dx_flat, axis=1)

    print(f"dx mean abs: {np.mean(np.abs(dx_flat), axis=0)}")
    print(f"dx max abs:  {np.max(np.abs(dx_flat), axis=0)}")
    print(f"dx norm max: {np.max(dx_norm):.6e}")

    print_worst_points(
        U_flat=U_flat,
        Y_flat=Y_flat,
        dx_flat=dx_flat,
        top_k=20,
    )

    # ------------------------------------------------------------
    # 3. Build bad physics mask
    # ------------------------------------------------------------
    good_mask_flat, bad_mask_flat = build_bad_physics_mask(
        dx_flat,
    )
    if np.any(bad_mask_flat):
        U_bad = U_flat[bad_mask_flat]

        u1_max_domain = U_flat[:, 0].max()
        u2_max_domain = U_flat[:, 1].max()

        # Candidate cutoffs only need to be tested at bad-point coordinates
        u1_candidates = np.unique(U_bad[:, 0])
        u2_candidates = np.unique(U_bad[:, 1])

        best_area = -np.inf
        best_pair = None

        for p1 in u1_candidates:
            for p2 in u2_candidates:
                # Is there any bad point strictly northeast of (p1, p2)?
                has_bad_NE = np.any((U_bad[:, 0] > p1) & (U_bad[:, 1] > p2))

                if not has_bad_NE:
                    area = (u1_max_domain - p1) * (u2_max_domain - p2)

                    if area > best_area:
                        best_area = area
                        best_pair = (p1, p2)

        p1, p2 = best_pair

        print(f"{well_name}: best cutoff pair = ({p1:.6f}, {p2:.6f})")
        print(f"Safe region: u1 > {p1:.6f} and u2 > {p2:.6f}")
        print(f"Safe NE area = {best_area:.6f}")

    else:
        print(f"{well_name}: no bad points")


    n_total = U_flat.shape[0]
    n_good = int(np.sum(good_mask_flat))
    n_bad = int(np.sum(bad_mask_flat))

    print("=" * 100)
    print(f"Total points: {n_total}")
    print(f"Good physics points: {n_good} ({100*n_good/n_total:.2f}%)")
    print(f"Bad physics points:  {n_bad} ({100*n_bad/n_total:.2f}%)")
    print("=" * 100)


    plot_physics_residual_maps(
        U1=U1,
        U2=U2,
        dx_flat=dx_flat,
        bad_mask_flat=bad_mask_flat,
        n_u1=n_u1,
        n_u2=n_u2,
    )