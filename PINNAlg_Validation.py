from Utilities.block_builders import *
from Networks.networks import PINN, AlgNN
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


def plot_3_surfaces(U1, U2, Z, save_dir: str = "Training", prefix: str = "algnn_sweep"):
    """
    Z[...,0] -> m_o_out
    Z[...,1] -> p_bh
    Z[...,2] -> p_tb_b
    """
    os.makedirs(save_dir, exist_ok=True)
    names = ["m_o_out", "p_bh", "p_tb_b"]

    for j, name in enumerate(names):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(U1, U2, Z[:, :, j], linewidth=0, antialiased=True)
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_zlabel(name)
        ax.set_title(f"{name}(u1,u2)")

        # --- Make (u1=1,u2=1) the closest vertex ---
        # ax.set_xlim(U1.max(), U1.min())
        ax.set_ylim(U2.max(), U2.min())
        ax.view_init(elev=25, azim=-60)

        out_path = os.path.join(save_dir, f"{prefix}_{name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths (adjust if yours are different) ---
    pinn_path = "Training/PINN.pth"
    algnn_path = "Training/AlgNN.pth"   # <-- change if needed

    pinn = rebuild_pinn_from_weights(pinn_path, device=device)
    algnn = rebuild_algnn_from_weights(algnn_path, device=device)

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

    # If you still want the y1,y2,y3 plots, you can reuse your earlier plot function on Y.
    plot_3_surfaces(U1, U2, Z, save_dir="Training", prefix="pinn_to_algnn")