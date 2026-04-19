from utilities.block_builders import *
from networks.networks import PINN, AlgNN
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def sweep_pinn_then_algnn(
    pinn: PINN,
    algnn: AlgNN,
    u1_min: float,
    u1_max: float,
    u2_min: float,
    u2_max: float,
    n_u1: int = 101,
    n_u2: int = 101,
    device: str = "cpu",
    batch_size: int = 16384,
):
    """
    Returns:
      U1, U2 : meshgrids shape (n_u2, n_u1)
      Y      : pinn outputs shape (n_u2, n_u1, 3)
      Z      : algnn outputs shape (n_u2, n_u1, 3) -> [m_o_out, p_bh, p_tb_b]
    """
    u1 = np.linspace(u1_min, u1_max, n_u1, dtype=np.float32)
    u2 = np.linspace(u2_min, u2_max, n_u2, dtype=np.float32)
    U1, U2 = np.meshgrid(u1, u2, indexing="xy")

    pts_u = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)  # (N,2)
    u_t = torch.from_numpy(pts_u).to(device)

    ys = []
    zs = []

    for k in range(0, u_t.shape[0], batch_size):
        ub = u_t[k : k + batch_size]               # (B,2)
        yb = pinn(ub)                              # (B,3)
        zb = algnn(yb, ub)                         # (B,3)
        ys.append(yb.detach().cpu())
        zs.append(zb.detach().cpu())

    Y = torch.cat(ys, dim=0).numpy().reshape(n_u2, n_u1, -1)
    Z = torch.cat(zs, dim=0).numpy().reshape(n_u2, n_u1, -1)
    return U1, U2, Y, Z


if __name__ == "__main__":
    from configuration.wells import get_wells
    from application.plotting_engine import plot_surface

    wells = get_wells()
    well_name = "P2"
    well_list = wells[well_name]

    y_min = well_list["y_min"]
    y_max = well_list["y_max"]
    z_min=well_list["z_min"]
    z_max=well_list["z_max"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths (adjust if yours are different) ---
    pinn_path = "../../well_models/well_2/PINN/PINN_P2.pth"
    algnn_path = "../../well_models/well_2/Alg/AlgNN_P2.pth"  # <-- change if needed

    pinn = rebuild_pinn_from_weights(pinn_path,
                                     y_min=y_min,
                                     y_max=y_max,
                                     device=device)

    algnn = rebuild_algnn_from_weights(algnn_path,
                                       y_min=y_min,
                                       y_max=y_max,
                                       z_min=z_min,
                                       z_max=z_max,
                                       device=device)

    # --- Sweep ranges ---
    u1_min, u1_max = 0.05, 1.0
    u2_min, u2_max = 0.10, 1.0

    U1, U2, Y, Z = sweep_pinn_then_algnn(
        pinn,
        algnn,
        u1_min=u1_min,
        u1_max=u1_max,
        u2_min=u2_min,
        u2_max=u2_max,
        n_u1=101,
        n_u2=101,
        device=device,
        batch_size=16384,
    )

    names = ["p_bh", "p_tb_b", "w_G_inj", "w_res"]
    for j, name in enumerate(names):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        plot_surface(
            fig=fig,
            ax=ax,
            U1=U1,
            U2=U2,
            Z=Z[:,:,j],
            title=f"{name}(u1,u2)",
            zlabel=f"y{j+1}",
        )
        plt.tight_layout()
        plt.show()

    #
    #
    # def plot_3_surfaces(U1, U2, Z, save_dir: str = "training", prefix: str = "algnn_sweep"):
    #     """
    #     Z[...,0] -> p_bh
    #     Z[...,1] -> p_tb_b
    #     """
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #
    #
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection="3d")
    #         ax.plot_surface(U1, U2, Z[:, :, j], linewidth=0, antialiased=True)
    #         ax.set_xlabel("u1")
    #         ax.set_ylabel("u2")
    #         ax.set_zlabel(name)
    #         ax.set_title(f"{name}(u1,u2)")
    #
    #         # --- Make (u1=1,u2=1) the closest vertex ---
    #         # ax.set_xlim(U1.max(), U1.min())
    #         ax.set_ylim(U2.max(), U2.min())
    #         ax.view_init(elev=25, azim=-60)
    #
    #         out_path = os.path.join(save_dir, f"{prefix}_{name}.png")
    #         plt.tight_layout()
    #         plt.savefig(out_path, dpi=200)
    #         plt.show()
    #         plt.close(fig)
    #         print(f"Saved: {out_path}")
    #
    #
    # # If you still want the y1,y2,y3 plots, you can reuse your earlier plot function on Y.
    # plot_3_surfaces(U1, U2, Z, save_dir="training", prefix="pinn_to_algnn")