import numpy as np

from application.simulation_engine import run_sweep
from application.simulation_engine import make_model

from configuration.wells import get_wells
def build_and_run_surrogate_sweep(
    u1_min: float,
    u2_min: float,
    N_data: int,
    y_guess_init,
    BSW: float,
    GOR: float,
    PI: float,
    K_gs: float,
    K_inj: float,
    K_pr: float
):
    """
    Creates the surrogate model, runs run_sweep on an N_data x N_data grid and get the results
    """

    model_sur = make_model("surrogate",
                           BSW=BSW,
                           GOR=GOR,
                           PI=PI,
                           K_gs=K_gs,
                           K_inj=K_inj,
                           K_pr=K_pr)

    results = run_sweep(
        model_sur,
        U1_MIN=u1_min,
        U2_MIN=u2_min,
        U_SIM_SIZE=N_data,
        y_guess_init=y_guess_init)
    return results


# def flatten_sweep_results_to_batch_full(results: dict, only_success: bool = True):
#     """
#     Like flatten_sweep_results_to_batch, but also returns z targets from OUT.
#
#     Assumes:
#       - first 3 Z_NAMES are states y = [y1,y2,y3]
#       - next 3 Z_NAMES are targets z = [p_bh_bar, p_tb_b_bar] (or whatever order is in Z_NAMES)
#     """
#     import numpy as np
#     import torch
#
#     u1_grid = np.asarray(results["u1_grid"], dtype=float)
#     u2_grid = np.asarray(results["u2_grid"], dtype=float)
#
#     Z_NAMES = list(results["Z_NAMES"])
#     OUT = results["OUT"]
#
#     SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
#     RES_DX = np.asarray(results["RES_DX"], dtype=float)
#
#     Nu1 = len(u1_grid)
#     Nu2 = len(u2_grid)
#
#     # u grid
#     U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
#     u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)
#
#     # y (first 3)
#     y_cols = []
#     for name in Z_NAMES[:3]:
#         arr = np.asarray(OUT[name], dtype=float)
#         y_cols.append(arr.reshape(-1))
#     y_flat = np.stack(y_cols, axis=1)
#
#     # z targets (next 3)
#     z_cols = []
#     names=["P_bh_bar", "P_tb_b_bar"]
#     for name in names:
#         arr = np.asarray(OUT[name], dtype=float)
#         z_cols.append(arr.reshape(-1))
#     z_flat = np.stack(z_cols, axis=1)
#
#     success_flat = SUCCESS.reshape(-1)
#     res_dx_flat = RES_DX.reshape(-1)
#
#     finite_y = np.all(np.isfinite(y_flat), axis=1)
#     finite_z = np.all(np.isfinite(z_flat), axis=1)
#     finite_all = finite_y & finite_z
#
#     mask = (success_flat & finite_all) if only_success else finite_all
#
#     u_np = u_flat[mask]
#     y_np = y_flat[mask]
#     z_np = z_flat[mask]
#     res_dx_np = res_dx_flat[mask]
#
#     # Torch tensors
#     u_t = torch.tensor(u_np, dtype=torch.float32)
#     y_t = torch.tensor(y_np, dtype=torch.float32)
#     z_t = torch.tensor(z_np, dtype=torch.float32)
#     res_dx_t = torch.tensor(res_dx_np, dtype=torch.float32)
#
#     return {
#         "Z_NAMES": Z_NAMES,
#         "u_np": u_np, "y_np": y_np, "z_np": z_np,
#         "u_t": u_t, "y_t": y_t, "z_t": z_t,
#         "res_dx_t": res_dx_t,
#         "Nu1": Nu1, "Nu2": Nu2,
#         "mask_np": mask,
#     }

def flatten_sweep_results_to_batch_full(results: dict, only_success: bool = True):
    """
    Like flatten_sweep_results_to_batch, but also returns z targets from OUT.

    Assumes:
      - first 3 Z_NAMES are states y = [y1,y2,y3]
      - z targets are selected explicitly by name
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

    # y (first 3)
    y_cols = []
    for name in Z_NAMES[:3]:
        arr = np.asarray(OUT[name], dtype=float)
        y_cols.append(arr.reshape(-1))
    y_flat = np.stack(y_cols, axis=1)

    # z targets
    z_names = ["P_bh_bar", "P_tb_b_bar","w_G_inj","w_res"]
    z_cols = []
    for name in z_names:
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
        "z_names": z_names,
        "u_np": u_np,
        "y_np": y_np,
        "z_np": z_np,
        "res_dx_np": res_dx_np,
        "u_t": u_t,
        "y_t": y_t,
        "z_t": z_t,
        "res_dx_t": res_dx_t,
        "Nu1": Nu1,
        "Nu2": Nu2,
        "mask_np": mask,
    }

if __name__ == "__main__":
    import numpy as np
    from configuration.wells import get_wells

    N_data = 20  # -> total grid points = 400

    wells = get_wells()
    well = "P2"

    BSW = wells[well]["BSW"]
    GOR = wells[well]["GOR"]
    PI = wells[well]["PI"]
    K_gs = wells[well]["K_gs_sur"]
    K_inj = wells[well]["K_inj_sur"]
    K_pr = wells[well]["K_pr_sur"]
    y_guess_sur = wells[well]["y_guess_sur"]

    u1_min, u1_max = 0.05, 1.00
    u2_min, u2_max = 0.10, 1.00

    y_guess_sur = np.array(y_guess_sur, dtype=float)

    print("Running sweep...")
    results = build_and_run_surrogate_sweep(
        u1_min=u1_min,
        u2_min=u2_min,
        N_data=N_data,
        y_guess_init=y_guess_sur,
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr
    )

    print("\nFlattening to AlgNN batch...")
    batch = flatten_sweep_results_to_batch_full(
        results,
        only_success=True
    )

    # -------- State ranges --------
    y_np = batch["y_np"]
    y_min = np.min(y_np, axis=0)
    y_max = np.max(y_np, axis=0)

    margin = 0.15
    y_span = y_max - y_min
    y_min_loose = y_min - margin * y_span
    y_max_loose = y_max + margin * y_span

    print("\n--- State ranges (loose) ---")
    for i, name in enumerate(batch["Z_NAMES"][:3]):
        print(f"{name}: min = {y_min_loose[i]:.6f}, max = {y_max_loose[i]:.6f}")

    # -------- Target ranges --------
    z_np = batch["z_np"]
    z_min = np.min(z_np, axis=0)
    z_max = np.max(z_np, axis=0)

    names=["P_bh_bar", "P_tb_b_bar","w_G_inj","w_res"]

    print("\n--- Target ranges ---")
    for i, name in enumerate(names):
        print(f"{name}: min = {z_min[i]:.6f}, max = {z_max[i]:.6f}")

    # -------- Summary --------
    print("\n--- Sweep summary ---")
    Nu1, Nu2 = batch["Nu1"], batch["Nu2"]
    total = Nu1 * Nu2
    success_total = int(np.sum(results["SUCCESS"]))
    print(f"Grid: {Nu1} x {Nu2} = {total} points")
    print(f"SUCCESS count (raw): {success_total}")
    print(f"Batch size (finite OUT & success): {batch['u_np'].shape[0]}")
    print(f"State names:  {batch['Z_NAMES'][:3]}")
    print(f"Target names: {batch['z_names']}")

    # -------- Tensor shapes --------
    print("\n--- Torch tensors ---")
    print("u_t:", tuple(batch["u_t"].shape), batch["u_t"].dtype)
    print("y_t:", tuple(batch["y_t"].shape), batch["y_t"].dtype)
    print("z_t:", tuple(batch["z_t"].shape), batch["z_t"].dtype)
    print("res_dx_t:", tuple(batch["res_dx_t"].shape), batch["res_dx_t"].dtype)

    # -------- Sample rows --------
    print("\n--- First samples ---")
    for k in range(min(1000, batch["u_np"].shape[0])):
        u_k = batch["u_np"][k]
        y_k = batch["y_np"][k]
        z_k = batch["z_np"][k]
        rd_k = batch["res_dx_np"][k]
        print(f"{k:02d} | u={u_k} | res_dx={rd_k:.2e} | y={y_k} | z={z_k}")


######################################################






def flatten_sweep_results_to_batch(results: dict,
                                   only_success: bool = True):
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
    U1=results["U1"]
    U2=results["U2"]

    # U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
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


# if __name__ == "__main__":
#     N_data = 20  # -> total grid points = 100
#
#     wells=get_wells()
#     well="P2"
#
#     BSW = wells[well]["BSW"]
#     GOR = wells[well]["GOR"]
#     PI = wells[well]["PI"]
#     K_gs = wells[well]["K_gs_sur"]
#     K_inj = wells[well]["K_inj_sur"]
#     K_pr = wells[well]["K_pr_sur"]
#     y_guess_sur = wells[well]["y_guess_sur"]
#
#     # Choose your bounds (edit these!)
#     u1_min, u1_max = 0.05, 1.00
#     u2_min, u2_max = 0.10, 1.00
#
#     # Your initial guess for the solver (edit if needed)
#     # Must match model nx (for your surrogate it's 3)
#
#     y_guess_sur = np.array(y_guess_sur, dtype=float)
#     RES_TOL_DX=1e-6
#     RES_TOL_G=1e-6
#     TOL_EIG=1e-8
#     print("Running sweep...")
#     results = build_and_run_surrogate_sweep(
#         u1_min=u1_min,
#         u2_min=u2_min,
#         N_data=N_data,
#         y_guess_init=y_guess_sur,
#         BSW=BSW,
#         GOR=GOR,
#         PI=PI,
#         K_gs=K_gs,
#         K_inj=K_inj,
#         K_pr=K_pr
#     )
#
#     print("\nFlattening to batch...")
#     batch = flatten_sweep_results_to_batch(results,
#                                            only_success=True)
#
#     device='cpu'
#
#     y_np = batch["y_np"]
#
#     y_min = np.min(y_np, axis=0)
#     y_max = np.max(y_np, axis=0)
#
#     margin = 0.15  # 15%
#
#     y_span = y_max - y_min
#     y_min_loose = y_min - margin * y_span
#     y_max_loose = y_max + margin * y_span
#
#     print("\n--- State ranges (loose) ---")
#     for i, name in enumerate(batch["Z_NAMES"][:3]):
#         print(f"{name}: min = {y_min_loose[i]:.6f}, max = {y_max_loose[i]:.6f}")
#
#
#     u = batch["u_t"].to(device)
#     y = batch["y_t"].to(device)
#     # z = batch["z_t"].to(device)  # <- ground truth for AlgNN
#     print("Z_NAMES:", batch["Z_NAMES"][:3])
#     print(u.shape, y.shape)
#
#     print("\n--- Sweep summary ---")
#     Nu1, Nu2 = batch["Nu1"], batch["Nu2"]
#     total = Nu1 * Nu2
#     success_total = int(np.sum(results["SUCCESS"]))
#     print(f"Grid: {Nu1} x {Nu2} = {total} points")
#     print(f"SUCCESS count (raw): {success_total}")
#     print(f"Batch size (finite OUT & success): {batch['u_np'].shape[0]}")
#     print(f"Z_NAMES: {batch['Z_NAMES']}")
#
#     # Torch shapes
#     print("\n--- Torch tensors ---")
#     print("u_t:", tuple(batch["u_t"].shape), batch["u_t"].dtype)
#     print("res_dx_t:", tuple(batch["res_dx_t"].shape), batch["res_dx_t"].dtype)
#
#     # Print a few samples
#     print("\n--- First all samples ---")
#     for k in range(batch["u_np"].shape[0]):
#         u_k = batch["u_np"][k]
#         rd_k = batch["res_dx_np"][k]
#         y_k = batch["y_np"][k]
#         print(f"{k:02d} | u={u_k}  | res_dx={rd_k:.2e} | y={y_k} ")

