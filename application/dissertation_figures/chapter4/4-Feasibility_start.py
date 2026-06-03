import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from configuration.wells import get_wells
from configuration.wells import get_wells
from application.simulation_engine import *
from application.plotting_engine import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from settings import *

from settings import *

from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator


def plot_feasible_region_pretty(
    u1_grid,
    u2_grid,
    bottomhole_pressure_values,
    tubing_pressure_values,
    p_tb_max,
    p_bh_min,
    instability_coef_dict,
    instability_side="above",
    n_fine=500,
    save_path=None,
):
    """
    Thesis-quality feasible region plot in the (u1, u2) plane.

    The original pressure fields may be coarse, e.g. 20 x 20.
    This function interpolates them to a finer grid for visualization.
    """

    # --------------------------------------------------
    # Convert inputs
    # --------------------------------------------------
    u1_grid = np.asarray(u1_grid, dtype=float).reshape(-1)
    u2_grid = np.asarray(u2_grid, dtype=float).reshape(-1)

    P_BH = np.asarray(bottomhole_pressure_values, dtype=float)
    P_TB = np.asarray(tubing_pressure_values, dtype=float)

    expected_shape = (len(u1_grid), len(u2_grid))

    if P_BH.shape != expected_shape:
        raise ValueError(
            f"bottomhole_pressure_values has shape {P_BH.shape}, "
            f"but expected {expected_shape}."
        )

    if P_TB.shape != expected_shape:
        raise ValueError(
            f"tubing_pressure_values has shape {P_TB.shape}, "
            f"but expected {expected_shape}."
        )

    # --------------------------------------------------
    # Fine plotting grid
    # --------------------------------------------------
    u1_fine = np.linspace(u1_grid.min(), u1_grid.max(), n_fine)
    u2_fine = np.linspace(u2_grid.min(), u2_grid.max(), n_fine)

    U1, U2 = np.meshgrid(u1_fine, u2_fine, indexing="ij")

    points_fine = np.column_stack([U1.ravel(), U2.ravel()])

    # --------------------------------------------------
    # Interpolate pressure fields
    # --------------------------------------------------
    interp_bh = RegularGridInterpolator(
        (u1_grid, u2_grid),
        P_BH,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_tb = RegularGridInterpolator(
        (u1_grid, u2_grid),
        P_TB,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    P_BH_fine = interp_bh(points_fine).reshape(U1.shape)
    P_TB_fine = interp_tb(points_fine).reshape(U1.shape)

    # --------------------------------------------------
    # Constraints
    # Feasible convention: G <= 0
    # --------------------------------------------------
    G_tb = P_TB_fine - p_tb_max
    G_bh = p_bh_min - P_BH_fine

    if "poly" in instability_coef_dict:
        poly = instability_coef_dict["poly"]
    elif "coef" in instability_coef_dict:
        poly = np.poly1d(instability_coef_dict["coef"])
    else:
        raise ValueError(
            "instability_coef_dict must contain either 'poly' or 'coef'."
        )

    U2_inst_boundary = poly(U1)

    if instability_side == "above":
        G_inst = U2_inst_boundary - U2
    elif instability_side == "below":
        G_inst = U2 - U2_inst_boundary
    else:
        raise ValueError("instability_side must be either 'above' or 'below'.")

    feasible = (
        (G_tb <= 0.0)
        & (G_bh <= 0.0)
        & (G_inst <= 0.0)
        & np.isfinite(G_tb)
        & np.isfinite(G_bh)
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    color_stable = (0.00, 0.45, 0.00)  # dark green
    color_unstable = (0.75, 0.00, 0.00)  # dark red
    color_boundary = (0.00, 0.20, 0.80) #blue

    # Trust-region center
    u_center = np.array([0.8, 0.8])
    radius = 0.1

    # Point at the center
    ax.scatter(
        u_center[0],
        u_center[1],
        s=55,
        color="black",
        marker="o",
        zorder=5,
        label="Current point",
    )

    # Circle centered at the point
    circle = Circle(
        xy=(u_center[0], u_center[1]),
        radius=radius,
        fill=False,
        color="black",
        linewidth=1.8,
        linestyle=":",
        zorder=4,
    )

    ax.add_patch(circle)

    # Label for the center point
    ax.text(
        u_center[0]-0.010,
        u_center[1] - 0.045,
        r"$\mathbf{u}_0$",
        fontsize=12,
        ha="left",
        va="bottom",
        zorder=6,
    )

    # Radius line from center to circumference
    u_radius_end = np.array([u_center[0] + radius, u_center[1]])

    ax.plot(
        [u_center[0], u_radius_end[0]],
        [u_center[1], u_radius_end[1]],
        color="black",
        linewidth=1.5,
        linestyle="-",
        zorder=5,
    )

    # Label for radius
    ax.text(
        u_center[0] + radius / 2,
        u_center[1] + 0.00025,
        r"$\Delta_0$",
        fontsize=12,
        ha="center",
        va="bottom",
        zorder=6,
    )

    # Soft feasible region
    ax.contourf(
        U1,
        U2,
        feasible.astype(float),
        levels=[0.5, 1.5],
        colors=color_stable,
        alpha=0.20,
        zorder=1
    )

    # Constraint boundaries
    ax.contour(
        U1,
        U2,
        G_tb,
        levels=[0.0],
        colors=color_boundary,
        linewidths=2.2,
        linestyles="solid",
        zorder=2
    )

    ax.contour(
        U1,
        U2,
        G_bh,
        levels=[0.0],
        colors=color_boundary,
        linewidths=2.2,
        linestyles="dashed",
        zorder=2
    )

    ax.contour(
        U1,
        U2,
        G_inst,
        levels=[0.0],
        colors=color_boundary,
        linewidths=2.0,
        linestyles="dashdot",
        zorder=2
    )

    # Optional: plot original grid points lightly to show data source
    # ax.scatter(U1[::20, ::20], U2[::20, ::20], s=3, color="k", alpha=0.15)

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")

    ax.set_xlim(u1_grid.min(), u1_grid.max())
    ax.set_ylim(u2_grid.min(), u2_grid.max())

    # ax.grid(False, alpha=0.18, linewidth=0.8)

    # Remove top/right visual heaviness
    # ax.spines["top"].set_alpha(0.5)
    # ax.spines["right"].set_alpha(0.5)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    legend_elements = [
        Patch(facecolor=color_stable, edgecolor="none", alpha=0.20,
              label="Feasible region"),
        Line2D([0], [0], color=color_boundary, lw=2.2, linestyle="solid",
               label=r"$p_{\mathrm{tb}}$ constraint"),
        Line2D([0], [0], color=color_boundary, lw=2.2, linestyle="dashed",
               label=r"$p_{\mathrm{bh}}$ constraint"),
        Line2D([0], [0], color=color_boundary, lw=2.0, linestyle="dashdot",
               label="Instability boundary"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()

    return fig, ax, feasible


if __name__ == "__main__":
    ##################################################################################3
    ######################################################################################
    wells = get_wells()
    well = "P2"
    params = wells[well]
    MODE = "rigorous"
    U1_MIN = 0.05  # if you ever change this, you need to change inside the black box optimizer
    U2_MIN = 0.10  # if you ever change this, you need to change inside the black box optimizer

    coeff_stability = {}

    print("\n" + "=" * 80)
    print(f"RUNNING WELL {well}")
    print("=" * 80)

    BSW = params["BSW"]
    GOR = params["GOR"]
    PI = params["PI"]
    K_gs = params["K_gs"]
    K_inj = params["K_inj"]
    K_pr = params["K_pr"]
    y_guess_rig = params["y_guess_rig"]
    z_guess_rig = params["z_guess_rig"]

    model_rig = make_model("rigorous",
                           BSW=BSW,
                           GOR=GOR,
                           PI=PI,
                           K_gs=K_gs,
                           K_inj=K_inj,
                           K_pr=K_pr)

    y_guess = np.array(y_guess_rig, dtype=float).reshape(-1)
    if y_guess.size != model_rig["nx"]:
        raise ValueError(f"y_guess_init has size {y_guess.size}, but model nx={model_rig["nx"]}")

    results = run_sweep(model_rig,
                        U1_MIN=U1_MIN,
                        U2_MIN=U2_MIN,
                        U_SIM_SIZE=U_SIM_SIZE,
                        y_guess_init=y_guess_rig,
                        z_guess_init=z_guess_rig,
                        RES_TOL_DX=RES_TOL_DX,
                        TOL_EIG=TOL_EIG)

    bottomhole_pressure_values = results["OUT"]["P_bh_bar"]
    tubing_pressure_values = results["OUT"]["P_tb_b_bar"]

    print(f"The u1_grid is {results["u1_grid"]}")
    print(f"The u2_grid is {results["u2_grid"]}")

    boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(
        results["u1_grid"],
        results["u2_grid"],
        results["STABLE"]
    )
    b_hat = fit_boundary_polynomial(boundary_u1,
                                    boundary_u2,
                                    deg=degree_polynomial)
    print(boundary_u1, boundary_u2)
    print(b_hat)
    coeff_stability = {
        "coef": b_hat.c,
        "poly": b_hat
    }

    print(f"My coefficients dictionary is {coeff_stability}")
    fig, ax, feasible = plot_feasible_region_pretty(
        u1_grid=results["u1_grid"],
        u2_grid=results["u2_grid"],
        bottomhole_pressure_values=bottomhole_pressure_values,
        tubing_pressure_values=tubing_pressure_values,
        p_tb_max=120.0,
        p_bh_min=80.0,
        instability_coef_dict=coeff_stability,
        instability_side="above",
        n_fine=500,
        save_path="feasible_region.pdf",
    )