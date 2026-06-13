from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from utilities.block_builders import *
from application.simulation_engine import evaluate_casadi_function_on_grid
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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



def plot_static_models_figure(
    well_name,
    variable_name,
    F_u2z_Poly,
    F_u2z_NN,
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
    # =================================================
    # 1. Import sweep from picke
    # =================================================

    if variable_name not in Z_NAMES:
        raise ValueError(f"variable_name must be one of {Z_NAMES}")

    var_idx = Z_NAMES.index(variable_name)
    folder = get_well_folder(well_name)
    sweep_path = folder / "sweep_results_figures_training.pkl"
    sweep = load_pickle(sweep_path)

    u1_grid = np.asarray(sweep["u1_grid"], dtype=float)
    u2_grid = np.asarray(sweep["u2_grid"], dtype=float)

    U1 = sweep["U1"]
    U2 = sweep["U2"]

    # Reference/sweep output
    Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)

    # =================================================
    # 2. Obtain poly and NN outputs and errors
    # =================================================

    # Polynomial output
    Z_poly = evaluate_casadi_function_on_grid(F_u2z_Poly, u1_grid, u2_grid)
    Y_poly = Z_poly[:, :, var_idx]

    # NN output
    Z_NN= evaluate_casadi_function_on_grid(F_u2z_NN, u1_grid, u2_grid)
    Y_NN = Z_NN[:, :, var_idx]

    # Absolute error Poly
    abs_error_Poly = np.abs(Y_poly - Y_sweep)

    # Absolute error NN
    abs_error_NN = np.abs(Y_NN - Y_sweep)


    # Use same color limits for sweep, poly and NN
    vmin_results = np.nanmin([np.nanmin(Y_sweep), np.nanmin(Y_poly),np.nanmin(Y_NN)])
    vmax_results = np.nanmax([np.nanmax(Y_sweep), np.nanmax(Y_poly),np.nanmax(Y_NN)])

    # Use same color limits for ERROR poly and NN
    vmin_error = np.nanmin([np.nanmin(abs_error_NN), np.nanmin(abs_error_Poly)])
    vmax_error = np.nanmax([np.nanmax(abs_error_NN), np.nanmax(abs_error_Poly)])

    # Shared normalizations
    norm_results = Normalize(vmin=vmin_results, vmax=vmax_results)
    norm_error = Normalize(vmin=vmin_error, vmax=vmax_error)

    fig, axes = plt.subplots(2, 3,
                             figsize=(7.2, 6.0),
                             constrained_layout=False)

    fig.subplots_adjust(
        left=0.10,
        right=0.95,
        top=0.93,
        bottom=0.22,
        wspace=0.45,
        hspace=0.55
    )

    for ax in axes.flat:
        ax.set_box_aspect(1)

    print(vmin_results)
    print(vmax_results)
    print(vmin_error)
    print(vmax_error)

    levels_results = np.linspace(vmin_results, vmax_results, 31)
    levels_error = np.linspace(vmin_error, vmax_error, 11)

    # =================================================
    # Row 1: Polynomial
    # =================================================




    c00 = axes[0, 0].contourf(U1, U2, Y_sweep, levels=levels_results, norm=norm_results)
    # axes[0, 0].contour(U1, U2, Y_sweep, levels=levels_results, linewidths=0.4, colors="k", alpha=0.45)
    axes[0, 0].set_title(r"r($\mathbf{u}$)")
    axes[0, 0].set_xlabel("$u_1$")
    axes[0, 0].set_ylabel("$u_2$")

    c01 = axes[0, 1].contourf(U1, U2, Y_poly, levels=levels_results, norm=norm_results)
    # axes[0, 1].contour(U1, U2, Y_poly, levels=levels_results, linewidths=0.4, colors="k", alpha=0.45)
    axes[0, 1].set_title(r"Poly $\tilde{r}(\mathbf{u})$")
    axes[0, 1].set_xlabel("$u_1$")
    axes[0, 1].set_ylabel("$u_2$")

    c02 = axes[0, 2].contourf(U1, U2, abs_error_Poly, levels=levels_error, norm=norm_error)
    axes[0, 2].set_title(r"$\|r(\mathbf{u})-\tilde{r}(\mathbf{u})\|$")
    axes[0, 2].set_xlabel("$u_1$")
    axes[0, 2].set_ylabel("$u_2$")

    # =================================================
    # Row 2: NN
    # =================================================
    c10 = axes[1, 0].contourf(U1, U2, Y_sweep, levels=levels_results, norm=norm_results)
    # axes[1, 0].contour(U1, U2, Y_sweep, levels=levels_results, linewidths=0.4, colors="k", alpha=0.45)
    axes[1, 0].set_title("r($\mathbf{u}$)")
    axes[1, 0].set_xlabel("$u_1$")
    axes[1, 0].set_ylabel("$u_2$")

    c11 = axes[1, 1].contourf(U1, U2, Y_NN, levels=levels_results, norm=norm_results)
    # axes[1, 1].contour(U1, U2, Y_NN, levels=levels_results, linewidths=0.4, colors="k", alpha=0.45)
    axes[1, 1].set_title(r"PINN-AlgNN $\tilde{r}(\mathbf{u})$")
    axes[1, 1].set_xlabel("$u_1$")
    axes[1, 1].set_ylabel("$u_2$")

    c12 = axes[1, 2].contourf(U1, U2, abs_error_NN, levels=levels_error, norm=norm_error)
    axes[1, 2].set_title(r"$\|r(\mathbf{u})-\tilde{r}(\mathbf{u})\|$")
    axes[1, 2].set_xlabel("$u_1$")
    axes[1, 2].set_ylabel("$u_2$")


    # =================================================
    # Shared horizontal colorbars
    # =================================================
    sm_results = ScalarMappable(norm=norm_results, cmap=c00.cmap)
    sm_results.set_array([])

    sm_error = ScalarMappable(norm=norm_error, cmap=c02.cmap)
    sm_error.set_array([])

    cax_results = fig.add_axes([0.10, 0.10, 0.53, 0.025])
    cbar_results = fig.colorbar(
        sm_results,
        # ax=axes[:, 0:2],
        cax=cax_results,
        orientation="horizontal",
    )
    cbar_results.set_label(r"Oil production $w_{o,out}$", labelpad=4)

    cax_error = fig.add_axes([0.73, 0.10, 0.23, 0.025])
    cbar_error = fig.colorbar(
        sm_error,
        # ax=axes[:, 2],
        cax=cax_error,
        orientation="horizontal",
    )

    cbar_error.set_label(r"Abs.error in $w_{o,out}$", labelpad=4)
    # fig.suptitle(f"{well_name} - {variable_name}", fontsize=12)

    fig.savefig(
        f"static_model_comparison_well{well_name}.pdf",
        bbox_inches="tight",
    )

    plt.show()

    rmse_poly = np.sqrt(np.nanmean((Y_poly - Y_sweep) ** 2))
    rmse_NN = np.sqrt(np.nanmean((Y_NN - Y_sweep) ** 2))



    return {
        "Y_sweep": Y_sweep,
        "Y_poly": Y_poly,
        "Y_NN": Y_NN,
        "rmse_poly": rmse_poly,
        "rmse_NN": rmse_NN,
    }


if __name__ == "__main__":

    well_name = "P2"
    model_NN = build_casadi_surrogate_u2z_for_well(well_name)
    model_poly = build_casadi_polynomial_u2z_for_well(well_name)


    result = plot_static_models_figure(
        well_name=well_name,
        variable_name="w_o_out",
        F_u2z_Poly=model_poly,
        F_u2z_NN=model_NN)

    print(f"RMSE poly for well {well_name} is:{result['rmse_poly']}")
    print(f"RMSE NN for well {well_name} is:{result['rmse_NN']}")