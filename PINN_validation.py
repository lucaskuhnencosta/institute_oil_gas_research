from Utilities.block_builders import *
from Networks.networks import PINN
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    U1, U2 = np.meshgrid(u1, u2, indexing="xy")

    pts = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)  # (N,2)
    pts_t = torch.from_numpy(pts).to(device)

    ys = []
    for k in range(0, pts_t.shape[0], batch_size):
        yb = model(pts_t[k : k + batch_size])
        ys.append(yb.detach().cpu())
    Y = torch.cat(ys, dim=0).numpy().reshape(n_u2, n_u1, -1)

    return U1, U2, Y


def plot_surfaces(U1, U2, Y, save_dir: str = "Training", prefix: str = "pinn_sweep"):
    """
    Makes 3 surface plots:
      plot 1: u1 x u2, y1
      plot 2: u1 x u2, y2
      plot 3: u1 x u2, y3
    Saves PNGs to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    # If you prefer contourf heatmaps instead of 3D surfaces, tell me and I’ll swap it.
    for j in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(U1, U2, Y[:, :, j], linewidth=0, antialiased=True)
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_zlabel(f"y{j+1}")
        ax.set_title(f"PINN sweep: y{j+1}(u1,u2)")

        ax.set_ylim(U2.max(), U2.min())
        ax.view_init(elev=25, azim=-60)

        out_path = os.path.join(save_dir, f"{prefix}_y{j+1}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.show()
        plt.close(fig)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "Training/PINN.pth"
    model = rebuild_pinn_from_weights(model_path, device=device)

    # ---- Choose sweep ranges here ----
    # Example: full normalized training range (you can change to any range you want)
    u1_min, u1_max = 0.05, 1.0
    u2_min, u2_max = 0.10, 1.0

    U1, U2, Y = sweep_pinn(
        model,
        u1_min=u1_min,
        u1_max=u1_max,
        u2_min=u2_min,
        u2_max=u2_max,
        n_u1=101,
        n_u2=101,
        device=device,
        batch_size=16384,
    )

    plot_surfaces(U1, U2, Y, save_dir="Training", prefix="pinn_sweep")