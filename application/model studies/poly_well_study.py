from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilities.block_builders import *


Z_NAMES = [
    "P_bh_bar",
    "P_tb_b_bar",
    "w_G_inj",
    "w_res",
    "w_L_res",
    "w_G_res",
    "w_w_out",
    "w_o_out",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_well_folder(well_name):
    return Path.cwd().parents[1] / "well_models" / str(well_name)


def evaluate_casadi_function_on_grid(F_u2z, u1_grid, u2_grid):
    """
    Evaluates F_u2z on the full sweep grid.

    Returns
    -------
    Z_poly : array, shape (Nu1, Nu2, 8)
    """

    Nu1 = len(u1_grid)
    Nu2 = len(u2_grid)

    Z_poly = np.full((Nu1, Nu2, 8), np.nan, dtype=float)

    for i, u1 in enumerate(u1_grid):
        for j, u2 in enumerate(u2_grid):
            z_val = F_u2z(np.array([u1, u2]))
            z_val = np.array(z_val, dtype=float).reshape(-1)

            Z_poly[i, j, :] = z_val

    return Z_poly


def plot_polynomial_vs_sweep(
    well_name,
    variable_name,
    F_u2z=None,
):
    """
    Plot sweep/reference, polynomial prediction, and absolute error
    for one well and one variable.

    Parameters
    ----------
    well_name : str
        Example: "P6"

    variable_name : str
        One of:
            P_bh_bar, P_tb_b_bar, w_G_inj, w_res,
            w_L_res, w_G_res, w_w_out, w_o_out

    F_u2z : casadi.Function or None
        If None, the function builds it using build_casadi_polynomial_u2z_for_well(well_name).
    """

    if variable_name not in Z_NAMES:
        raise ValueError(f"variable_name must be one of {Z_NAMES}")

    if F_u2z is None:
        F_u2z = build_casadi_polynomial_u2z_for_well(well_name)

    var_idx = Z_NAMES.index(variable_name)


    folder = get_well_folder(well_name)
    sweep_path = folder / "sweep_results.pkl"

    sweep = load_pickle(sweep_path)

    u1_grid = np.asarray(sweep["u1_grid"], dtype=float)
    u2_grid = np.asarray(sweep["u2_grid"], dtype=float)

    U1 = sweep["U1"]
    U2 = sweep["U2"]

    # Reference/sweep output
    Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)

    # Polynomial output
    Z_poly = evaluate_casadi_function_on_grid(F_u2z, u1_grid, u2_grid)
    Y_poly = Z_poly[:, :, var_idx]

    # Absolute error
    abs_error = np.abs(Y_poly - Y_sweep)

    # Use same color limits for sweep and polynomial
    vmin = np.nanmin([np.nanmin(Y_sweep), np.nanmin(Y_poly)])
    vmax = np.nanmax([np.nanmax(Y_sweep), np.nanmax(Y_poly)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    c0 = axes[0].contourf(U1, U2, Y_sweep, levels=30, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{well_name} - Sweep: {variable_name}")
    axes[0].set_xlabel("$u_1$")
    axes[0].set_ylabel("$u_2$")
    fig.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(U1, U2, Y_poly, levels=30, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{well_name} - Polynomial: {variable_name}")
    axes[1].set_xlabel("$u_1$")
    axes[1].set_ylabel("$u_2$")
    fig.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(U1, U2, abs_error, levels=30)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("$u_1$")
    axes[2].set_ylabel("$u_2$")
    fig.colorbar(c2, ax=axes[2])

    rmse = np.sqrt(np.nanmean((Y_poly - Y_sweep) ** 2))
    max_abs = np.nanmax(abs_error)

    fig.suptitle(
        f"{well_name} | {variable_name} | RMSE = {rmse:.4e}, MaxAbs = {max_abs:.4e}",
        fontsize=12,
    )

    plt.show()

    return {
        "Y_sweep": Y_sweep,
        "Y_poly": Y_poly,
        "abs_error": abs_error,
        "rmse": rmse,
        "max_abs_error": max_abs,
    }

result = plot_polynomial_vs_sweep(
    well_name="P6",
    variable_name="w_o_out",
)

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import numpy as np
import matplotlib.pyplot as plt


def plot_polynomial_vs_sweep_3d(
    well_name,
    variable_name,
    F_u2z=None,
    elev=25,
    azim=-135,
):
    """
    3D comparison between sweep/reference, polynomial prediction,
    and absolute error for one well and one variable.
    """

    if variable_name not in Z_NAMES:
        raise ValueError(f"variable_name must be one of {Z_NAMES}")

    if F_u2z is None:
        F_u2z = build_casadi_polynomial_u2z_for_well(well_name)

    var_idx = Z_NAMES.index(variable_name)

    folder = get_well_folder(well_name)
    sweep_path = folder / "sweep_results.pkl"

    sweep = load_pickle(sweep_path)

    u1_grid = np.asarray(sweep["u1_grid"], dtype=float)
    u2_grid = np.asarray(sweep["u2_grid"], dtype=float)

    U1 = np.asarray(sweep["U1"], dtype=float)
    U2 = np.asarray(sweep["U2"], dtype=float)

    Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)

    Z_poly = evaluate_casadi_function_on_grid(F_u2z, u1_grid, u2_grid)
    Y_poly = Z_poly[:, :, var_idx]

    abs_error = np.abs(Y_poly - Y_sweep)

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(U1, U2, Y_sweep, linewidth=0, antialiased=True)
    ax1.set_title(f"{well_name} - Sweep: {variable_name}")
    ax1.set_xlabel("$u_1$")
    ax1.set_ylabel("$u_2$")
    ax1.set_zlabel(variable_name)
    ax1.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(U1, U2, Y_poly, linewidth=0, antialiased=True)
    ax2.set_title(f"{well_name} - Polynomial: {variable_name}")
    ax2.set_xlabel("$u_1$")
    ax2.set_ylabel("$u_2$")
    ax2.set_zlabel(variable_name)
    ax2.view_init(elev=elev, azim=azim)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(U1, U2, abs_error, linewidth=0, antialiased=True)
    ax3.set_title("Absolute error")
    ax3.set_xlabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_zlabel("|error|")
    ax3.view_init(elev=elev, azim=azim)

    rmse = np.sqrt(np.nanmean((Y_poly - Y_sweep) ** 2))
    max_abs = np.nanmax(abs_error)

    fig.suptitle(
        f"{well_name} | {variable_name} | RMSE = {rmse:.4e}, MaxAbs = {max_abs:.4e}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()

    return {
        "Y_sweep": Y_sweep,
        "Y_poly": Y_poly,
        "abs_error": abs_error,
        "rmse": rmse,
        "max_abs_error": max_abs,
    }

result_3d = plot_polynomial_vs_sweep_3d(
    well_name="P6",
    variable_name="w_G_inj",
    F_u2z=None,
    elev=25,
)
