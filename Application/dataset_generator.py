import numpy as np

from Application.model_analysis_application import run_sweep
from Application.model_analysis_application import make_model

def flatten_sweep_results_to_batch_full(results: dict, only_success: bool = True):
    """
    Like flatten_sweep_results_to_batch, but also returns z targets from OUT.

    Assumes:
      - first 3 Z_NAMES are states y = [y1,y2,y3]
      - next 3 Z_NAMES are targets z = [w_o_out, p_bh_bar, p_tb_b_bar] (or whatever order is in Z_NAMES)
    """
    import numpy as np
    import torch

    u1_grid = np.asarray(results["u1_grid"], dtype=float)
    u2_grid = np.asarray(results["u2_grid"], dtype=float)

    Z_NAMES = list(results["Z_NAMES"])
    OUT = results["OUT"]

    SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
    RES_DX = np.asarray(results["RES_DX"], dtype=float)

    Nu1 = len(u1_grid)
    Nu2 = len(u2_grid)

    # u grid
    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
    u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)

    if len(Z_NAMES) < 6:
        raise ValueError(f"Expected at least 6 Z_NAMES entries (3 y + 3 z), got {len(Z_NAMES)}")

    # y (first 3)
    y_cols = []
    for name in Z_NAMES[:3]:
        arr = np.asarray(OUT[name], dtype=float)
        y_cols.append(arr.reshape(-1))
    y_flat = np.stack(y_cols, axis=1)

    # z targets (next 3)
    z_cols = []
    for name in Z_NAMES[3:6]:
        arr = np.asarray(OUT[name], dtype=float)
        z_cols.append(arr.reshape(-1))
    z_flat = np.stack(z_cols, axis=1)

    success_flat = SUCCESS.reshape(-1)
    res_dx_flat = RES_DX.reshape(-1)

    finite_y = np.all(np.isfinite(y_flat), axis=1)
    finite_z = np.all(np.isfinite(z_flat), axis=1)
    finite_all = finite_y & finite_z

    mask = (success_flat & finite_all) if only_success else finite_all

    u_np = u_flat[mask]
    y_np = y_flat[mask]
    z_np = z_flat[mask]
    res_dx_np = res_dx_flat[mask]

    # Torch tensors
    u_t = torch.tensor(u_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    z_t = torch.tensor(z_np, dtype=torch.float32)
    res_dx_t = torch.tensor(res_dx_np, dtype=torch.float32)

    return {
        "Z_NAMES": Z_NAMES,
        "u_np": u_np, "y_np": y_np, "z_np": z_np,
        "u_t": u_t, "y_t": y_t, "z_t": z_t,
        "res_dx_t": res_dx_t,
        "Nu1": Nu1, "Nu2": Nu2,
        "mask_np": mask,
    }

def flatten_sweep_results_to_batch(results: dict, only_success: bool = True):
    """
    Convert run_sweep() output dict into a flat batch.

    Uses ONLY:
      - u = [u1, u2]
      - y = first 3 entries of Z_NAMES (assumed states)
      - success mask (stable is ignored)

    Returns:
      u_np : (N,2)
      y_np : (N,3)
      success_np : (N,) bool
      res_dx_np : (N,) float
      u_t : torch.FloatTensor (N,2)
      y_t : torch.FloatTensor (N,3)
      res_dx_t : torch.FloatTensor (N,)
    """
    import numpy as np
    import torch

    u1_grid = np.asarray(results["u1_grid"], dtype=float)
    u2_grid = np.asarray(results["u2_grid"], dtype=float)

    Z_NAMES = list(results["Z_NAMES"])
    OUT = results["OUT"]

    SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
    RES_DX = np.asarray(results["RES_DX"], dtype=float)

    Nu1 = len(u1_grid)
    Nu2 = len(u2_grid)

    # Build u grid (Nu1,Nu2) -> (Nu1*Nu2,2)
    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
    u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)

    # Extract ONLY first 3 outputs (states) in Z_NAMES order
    if len(Z_NAMES) < 3:
        raise ValueError(f"Expected at least 3 Z_NAMES entries, got {len(Z_NAMES)}")

    y_cols = []
    for name in Z_NAMES[:3]:
        arr = np.asarray(OUT[name], dtype=float)   # (Nu1,Nu2)
        y_cols.append(arr.reshape(-1))
    y_flat = np.stack(y_cols, axis=1)  # (Nu1*Nu2, 3)

    success_flat = SUCCESS.reshape(-1)
    res_dx_flat = RES_DX.reshape(-1)

    # Valid rows: success AND finite y
    finite_y = np.all(np.isfinite(y_flat), axis=1)
    mask = success_flat & finite_y if only_success else finite_y

    u_np = u_flat[mask]
    y_np = y_flat[mask]
    success_np = success_flat[mask]
    res_dx_np = res_dx_flat[mask]

    # Torch tensors
    u_t = torch.tensor(u_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    res_dx_t = torch.tensor(res_dx_np, dtype=torch.float32)

    return {
        "Z_NAMES": Z_NAMES,
        "u_np": u_np,
        "y_np": y_np,
        "success_np": success_np,
        "res_dx_np": res_dx_np,
        "u_t": u_t,
        "y_t": y_t,
        "res_dx_t": res_dx_t,
        "Nu1": Nu1,
        "Nu2": Nu2,
        "mask_np": mask,
    }

def build_and_run_surrogate_sweep(
    u1_min: float,
    u1_max: float,
    u2_min: float,
    u2_max: float,
    N_data: int,
    y_guess_init,
    BSW: float = 0.20,
    GOR: float = 0.05,
    PI: float = 3.0e-6,
):
    """
    Creates the surrogate model, runs run_sweep on an N_data x N_data grid.
    """
    # Grid (N_data is sqrt of number of points)
    u1_grid = np.linspace(u1_min, u1_max + 1e-5, N_data)
    u2_grid = np.linspace(u2_min, u2_max + 1e-5, N_data)

    model_sur = make_model("surrogate", BSW=BSW, GOR=GOR, PI=PI)

    results = run_sweep(
        model_sur,
        u1_grid=u1_grid,
        u2_grid=u2_grid,
        y_guess_init=y_guess_init,
        z_guess_init=None,
    )
    return results

if __name__ == "__main__":
    # ----------------------------
    # Debug run (as requested)
    # ----------------------------
    N_data = 10  # -> total grid points = 100

    # Choose your bounds (edit these!)
    u1_min, u1_max = 0.05, 1.0
    u2_min, u2_max = 0.10, 1.0

    # Your initial guess for the solver (edit if needed)
    # Must match model nx (for your surrogate it's 3)
    y_guess_sur = np.array([3285.42, 300.822, 6910.91], dtype=float)
    RES_TOL_DX=1e-6
    RES_TOL_G=1e-6
    TOL_EIG=1e-8
    print("Running sweep...")
    results = build_and_run_surrogate_sweep(
        u1_min=u1_min,
        u1_max=u1_max,
        u2_min=u2_min,
        u2_max=u2_max,
        N_data=N_data,
        y_guess_init=y_guess_sur,
        BSW=0.20,
        GOR=0.05,
        PI=3.0e-6,
    )

    print("\nFlattening to batch...")
    batch = flatten_sweep_results_to_batch_full(results, only_success=True)

    device='cpu'

    u = batch["u_t"].to(device)
    y = batch["y_t"].to(device)
    z = batch["z_t"].to(device)  # <- ground truth for AlgNN
    print("Z_NAMES:", batch["Z_NAMES"][:6])
    print(u.shape, y.shape, z.shape)
    #
    # print("\n--- Sweep summary ---")
    # Nu1, Nu2 = batch["Nu1"], batch["Nu2"]
    # total = Nu1 * Nu2
    # success_total = int(np.sum(results["SUCCESS"]))
    # print(f"Grid: {Nu1} x {Nu2} = {total} points")
    # print(f"SUCCESS count (raw): {success_total}")
    # print(f"Batch size (finite OUT & success): {batch['u_np'].shape[0]}")
    # print(f"Z_NAMES: {batch['Z_NAMES']}")
    #
    # # Torch shapes
    # print("\n--- Torch tensors ---")
    # print("u_t:", tuple(batch["u_t"].shape), batch["u_t"].dtype)
    # # print("z_t:", tuple(batch["z_t"].shape), batch["z_t"].dtype)
    # # print("stable_t:", tuple(batch["stable_t"].shape), batch["stable_t"].dtype)
    # print("res_dx_t:", tuple(batch["res_dx_t"].shape), batch["res_dx_t"].dtype)
    #
    # # Print a few samples
    # print("\n--- First 50 samples ---")
    # for k in range(min(50, batch["u_np"].shape[0])):
    #     u_k = batch["u_np"][k]
    #     rd_k = batch["res_dx_np"][k]
    #     y_k = batch["y_np"][k]
    #     print(f"{k:02d} | u={u_k}  | res_dx={rd_k:.2e} | y={y_k} ")
