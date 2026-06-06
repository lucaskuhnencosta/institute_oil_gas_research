from pandas.core.arrays import arrow

from settings import *
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import numpy as np
from configuration.wells import get_wells
from application.simulation_engine import make_model, run_sweep
from utilities.block_builders import build_casadi_surrogate_u2z_for_well
from application.plotting_engine import *
import matplotlib as mpl
from optimization.api_interface_simulator import plant_zeroth_and_first_order

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import casadi as ca
import numpy as np

def build_corrected_surrogate_single_with_plant_api(
    F_surr,
    plant_model,
    u0,
    y_guess,
    z_guess,
    out_names,
    well_name="well",
):
    """
    Build first-order corrected surrogate:

        z_corr(u) = z_surr(u)
                  + c
                  + C (u - u0)

    where

        c = z_plant(u0) - z_surr(u0)
        C = J_plant(u0) - J_surr(u0)

    The plant zeroth- and first-order information is obtained from
    plant_zeroth_and_first_order(...).
    """

    u0 = np.asarray(u0, dtype=float).reshape(2)

    # --------------------------------------------------
    # 1) Plant zeroth- and first-order information
    # --------------------------------------------------
    plant_eval = plant_zeroth_and_first_order(
        model=plant_model,
        u_k=u0,
        y_guess=y_guess,
        z_guess=z_guess,
        out_names=out_names,
    )

    z_p = np.asarray(plant_eval["z0"], dtype=float).reshape(-1)
    J_p = np.asarray(plant_eval["J"], dtype=float)

    print(f" z_p are{z_p}")
    print(f" J_p are{J_p}")

    # --------------------------------------------------
    # 2) Surrogate zeroth- and first-order information
    # --------------------------------------------------
    u_sym = ca.MX.sym(f"u_surr_eval_{well_name}", 2)

    z_surr_sym = F_surr(u=u_sym)["z"]
    J_surr_sym = ca.jacobian(z_surr_sym, u_sym)

    F_surr_eval = ca.Function(
        f"F_surr_eval_{well_name}",
        [u_sym],
        [z_surr_sym, J_surr_sym],
        ["u"],
        ["z", "J"],
    )

    surr_eval = F_surr_eval(u=u0)

    z_s = np.asarray(surr_eval["z"], dtype=float).reshape(-1)
    J_s = np.asarray(surr_eval["J"], dtype=float)

    print(f" z_s are{z_s}")
    print(f" J_s are{J_s}")

    # --------------------------------------------------
    # 3) Important: make sure dimensions match
    # --------------------------------------------------
    if z_p.shape[0] != z_s.shape[0]:
        raise ValueError(
            f"Dimension mismatch: plant z has length {z_p.shape[0]}, "
            f"but surrogate z has length {z_s.shape[0]}. "
            f"Check out_names."
        )

    if J_p.shape != J_s.shape:
        raise ValueError(
            f"Jacobian mismatch: plant J has shape {J_p.shape}, "
            f"but surrogate J has shape {J_s.shape}. "
            f"Check out_names and surrogate output ordering."
        )

    # --------------------------------------------------
    # 4) Correction terms
    # --------------------------------------------------
    c = z_p - z_s
    C = J_p - J_s

    print(f"corrected terms c are{c}")
    print(f"corrected terms C are{C}")

    # --------------------------------------------------
    # 5) Build corrected CasADi model
    # --------------------------------------------------
    u_corr = ca.MX.sym(f"u_corr_{well_name}", 2)

    z_surr_corr = F_surr(u=u_corr)["z"]
    du = u_corr - ca.DM(u0)

    z_corr = z_surr_corr + ca.DM(c) + ca.DM(C) @ du

    F_corr = ca.Function(
        f"F_corr_{well_name}",
        [u_corr],
        [z_corr],
        ["u"],
        ["z"],
    )

    # --------------------------------------------------
    # 6) Checks
    # --------------------------------------------------
    z_corr_u0 = np.asarray(F_corr(u=u0)["z"], dtype=float).reshape(-1)

    J_corr_sym = ca.jacobian(z_corr, u_corr)

    F_corr_eval = ca.Function(
        f"F_corr_eval_{well_name}",
        [u_corr],
        [z_corr, J_corr_sym],
        ["u"],
        ["z", "J"],
    )

    corr_eval = F_corr_eval(u=u0)
    J_corr_u0 = np.asarray(corr_eval["J"], dtype=float)

    debug = {
        "u0": u0,
        "z_p": z_p,
        "z_s": z_s,
        "J_p": J_p,
        "J_s": J_s,
        "c": c,
        "C": C,
        "check_zeroth": z_corr_u0 - z_p,
        "check_first": J_corr_u0 - J_p,
        "plant_eval": plant_eval,
    }

    return F_corr, debug


def plot_surrogate_vs_rigorous_contours(
    u1_grid,
    u2_grid,
    w_o_out_surr,
    w_o_out_rig,
    w_o_out_corr,
    u0=np.array([0.5, 0.5]),
    radius=0.3,
    n_levels=100,
    save_path=None,
):
    """
    Plot contour maps of w_o_out for:
        1) surrogate model
        2) rigorous model
        3) empty bottom panel for future corrected model

    Uses the project utility plot_contour(...).
    Only contour lines are plotted.
    """

    # --------------------------------------------------
    # Convert inputs
    # --------------------------------------------------
    u1_grid = np.asarray(u1_grid, dtype=float).reshape(-1)
    u2_grid = np.asarray(u2_grid, dtype=float).reshape(-1)

    W_surr = np.asarray(w_o_out_surr, dtype=float)
    W_rig = np.asarray(w_o_out_rig, dtype=float)
    W_corr = np.asarray(w_o_out_corr, dtype=float)


    expected_shape = (len(u1_grid), len(u2_grid))

    if W_surr.shape != expected_shape:
        raise ValueError(
            f"w_o_out_surr has shape {W_surr.shape}, expected {expected_shape}."
        )

    if W_rig.shape != expected_shape:
        raise ValueError(
            f"w_o_out_rig has shape {W_rig.shape}, expected {expected_shape}."
        )




    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")

    # --------------------------------------------------
    # Shared contour scale
    # --------------------------------------------------
    w_min = min(np.nanmin(W_surr), np.nanmin(W_rig),np.nanmin(W_corr))
    w_max = max(np.nanmax(W_surr), np.nanmax(W_rig),np.nanmax(W_corr))

    levels = np.linspace(w_min, w_max, n_levels)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    fig = plt.figure(figsize=(6.6,7.2))

    gs = gridspec.GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0 ,0.25, 1.0, 1.0,  0.0001, 0.10],
        height_ratios=[1.0, 1.50],
        wspace=0.20,
        hspace=0.45,
    )

    ax_surr = fig.add_subplot(gs[0, 0:2])
    ax_rig = fig.add_subplot(gs[0, 3:5])
    ax_bottom = fig.add_subplot(gs[1, 1:4])
    cax = fig.add_subplot(gs[:, 6])

    ax_surr.set_aspect("equal", adjustable="box")
    ax_rig.set_aspect("equal", adjustable="box")
    ax_bottom.set_aspect("equal", adjustable="box")

    # --------------------------------------------------
    # Helper: add u0 and trust-region radius
    # --------------------------------------------------
    def add_trust_region_annotation(ax,
                                    color,
                                    arrow_direction):
        ax.scatter(
            u0[0],
            u0[1],
            s=35,
            color=color,
            edgecolor="white" if color == "black" else "black",
            linewidth=0.9,
            zorder=5,
        )

        circle = Circle(
            xy=(u0[0], u0[1]),
            radius=radius,
            fill=False,
            color=color,
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        ax.add_patch(circle)

        ax.plot(
            [u0[0], u0[0] - radius],
            [u0[1], u0[1]],
            color=color,
            linewidth=1.3,
            zorder=5,
        )

        ax.text(
            u0[0] -0.05,
            u0[1] - 0.060,
            r"$\mathbf{u}_k$",
            fontsize=10,
            color=color,
            weight="bold",
            zorder=6,
        )

        ax.text(
            u0[0] - radius / 2,
            u0[1] + 0.00025,
            r"$\Delta_k$",
            fontsize=10,
            color=color,
            ha="center",
            va="bottom",
            weight="bold",
            zorder=6,
        )

        # ---------------------------------------------
        # Step direction arrow from u_k
        # ---------------------------------------------
        arrow_length = 0.15

        if arrow_direction == "east":
            du = arrow_length
            dv = 0.0
        elif arrow_direction == "ne":
            du = arrow_length / np.sqrt(2)
            dv = arrow_length / np.sqrt(2)
        else:
            raise ValueError("arrow_direction must be either 'east' or 'ne'.")

        ax.annotate(
            "",
            xy=(u0[0] + du, u0[1] + dv),
            xytext=(u0[0], u0[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="blue",
                linewidth=2.0,
                shrinkA=4,
                shrinkB=0,
            ),
            zorder=7,
        )

    # --------------------------------------------------
    # Helper: draw one panel using your plot_contour utility
    # --------------------------------------------------
    def draw_panel(ax,
                   W,
                   title,
                   arrow_direction,):
        ax, mappable = plot_contour(
            U1=U1,
            U2=U2,
            Z=W,
            title=title,
            zlabel=r"$Oil production rate, w_{o,\mathrm{out}}$",
            u1_opt=None,
            u2_opt=None,
            z_opt=None,
            line_levels=levels,
            fill_levels=levels,
            mark_optimum=True,
            ax=ax,
            add_colorbar=False,
            vmin=w_min,
            vmax=w_max,
            just_contour=True,
            cmap="viridis",
            linewidths=1.4,
            alpha=0.95,
            equal_aspect=True,
            xlim=(u1_grid.min(), u1_grid.max()),
            ylim=(u2_grid.min(), u2_grid.max()),
            xlabel=r"$u_1$",
            ylabel=r"$u_2$",
            # xticks=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
            # yticks=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        )

        add_trust_region_annotation(ax,
                                    color="black",
                                    arrow_direction=arrow_direction)

        return mappable

    # --------------------------------------------------
    # Top row
    # --------------------------------------------------
    mappable_surr = draw_panel(
        ax_surr,
        W_surr,
        r"$\tilde{r}(\mathbf{u})=n_{\theta}(\mathbf{u})$",
        arrow_direction="east"
    )

    mappable_rig = draw_panel(
        ax_rig,
        W_rig,
        r"$s(\mathbf{u})$",
        arrow_direction="ne"
    )

    mappable_corr = draw_panel(
        ax_bottom,
        W_corr,
        r"$r_k(\mathbf{u})$",
        arrow_direction="ne"
    )


    # ax_bottom.set_title(r"$r_k(\mathbf{u})$", pad=10)
    ax_bottom.set_xlabel(r"$u_1$")
    ax_bottom.set_ylabel(r"$u_2$")
    ax_bottom.set_aspect("equal", adjustable="box")

    ax_bottom.set_xlim(u1_grid.min(), u1_grid.max())
    ax_bottom.set_ylim(u2_grid.min(), u2_grid.max())

    ax_bottom.tick_params(axis="both", which="major", labelsize=12)
    ax_bottom.grid(True, alpha=0.18, linewidth=0.7)

    # add_trust_region_annotation(ax_bottom,
    #                             color="black")

    # --------------------------------------------------
    # Shared colorbar for contour lines
    # # --------------------------------------------------
    norm = mpl.colors.Normalize(vmin=w_min, vmax=w_max)

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap="viridis",
    )

    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"Oil production rate, $w_{o,\mathrm{out}}$")
    cb.ax.tick_params()


    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()

    return fig, (ax_surr, ax_rig, ax_bottom)



if "__main__" == __name__:
    # Normalize RGB (divide by 255)
    color_primary = (54 / 255, 32 / 255, 229 / 255)  # blue
    color_secondary = (240 / 255, 101 / 255, 74 / 255)  # orange
    color_third = (183 / 255, 53 / 255, 192 / 255)  # purple (saved)

    U1_MIN = 0.05  # if you ever change this, you need to change inside the black box optimizer
    U2_MIN = 0.10  # i
    model_type = "rigorous"
    wells = get_wells()
    well_names = list(wells.keys())

    rigorous_models = []
    F_u2z_models = []
    well_name="P5"
    well_data = wells[well_name]

    BSW = well_data["BSW"]
    GOR = well_data["GOR"]
    PI = well_data["PI"]
    K_gs = well_data["K_gs"]
    K_inj = well_data["K_inj"]
    K_pr = well_data["K_pr"]

    # RIGOROUS MODEL
    model_rig = make_model(
    model_type,
    BSW=BSW,
    GOR=GOR,
    PI=PI,
    K_gs=K_gs,
    K_inj=K_inj,
    K_pr=K_pr
    )
    y_guess_rig = well_data["y_guess_rig"]
    z_guess_rig = well_data["z_guess_rig"]

    results = run_sweep(model_rig,
                    U1_MIN=U1_MIN,
                    U2_MIN=U2_MIN,
                    U_SIM_SIZE=U_SIM_SIZE,
                    y_guess_init=y_guess_rig,
                    z_guess_init=z_guess_rig,
                    RES_TOL_DX=RES_TOL_DX,
                    TOL_EIG=TOL_EIG)
    w_o_out_rig = results["OUT"]["w_o_out"]


    # --------------------------------------------------
    # SURROGATE MODEL
    # --------------------------------------------------
    F_u2z = build_casadi_surrogate_u2z_for_well(well_name)

    u1_grid = np.linspace(U1_MIN, 1.00001, U_SIM_SIZE)
    u2_grid = np.linspace(U2_MIN, 1.00001, U_SIM_SIZE)

    w_o_out_surr = np.zeros((U_SIM_SIZE, U_SIM_SIZE))

    for i, u1 in enumerate(u1_grid):
        for j, u2 in enumerate(u2_grid):
            u_val = np.array([u1, u2], dtype=float)
            out = F_u2z(u=u_val)
            z_val = np.array(out["z"], dtype=float).reshape(-1)
            w_o_out_surr[i, j] = z_val[7]

    print(f"My final w_o_out_rig is {w_o_out_rig}")
    print(f"My final w_o_out_surr is {w_o_out_surr}")

    # --------------------------------------------------
    # CORRECTED MODEL
    # --------------------------------------------------
    u0 = np.array([0.5, 0.4], dtype=float)

    out_names_corr = [
        "P_bh_bar",
        "P_tb_b_bar",
        "w_G_inj",
        "w_res",
        "w_L_res",
        "w_G_res",
        "w_w_out",
        "w_o_out",
    ]

    F_corr, corr_debug = build_corrected_surrogate_single_with_plant_api(
        F_surr=F_u2z,
        plant_model=model_rig,
        u0=u0,
        y_guess=y_guess_rig,
        z_guess=z_guess_rig,
        out_names=out_names_corr,
        well_name=well_name,
    )

    w_o_out_corr = np.zeros((U_SIM_SIZE, U_SIM_SIZE))

    for i, u1 in enumerate(u1_grid):
        for j, u2 in enumerate(u2_grid):
            u_val = np.array([u1, u2], dtype=float)

            out = F_corr(u=u_val)

            z_val = np.array(out["z"], dtype=float).reshape(-1)

            w_o_out_corr[i, j] = z_val[7]

    print("Correction point:", corr_debug["u0"])
    print("||c||:", np.linalg.norm(corr_debug["c"]))
    print("||C||:", np.linalg.norm(corr_debug["C"]))
    print("Zeroth-order check:", np.linalg.norm(corr_debug["check_zeroth"]))
    print("First-order check:", np.linalg.norm(corr_debug["check_first"]))

    fig, axes = plot_surrogate_vs_rigorous_contours(
    u1_grid=u1_grid,
    u2_grid=u2_grid,
    w_o_out_surr=w_o_out_surr,
    w_o_out_rig=w_o_out_rig,
    w_o_out_corr=w_o_out_corr,
    u0=u0,
    radius=0.15,
    n_levels=90,
    save_path=f"wo_out_surr_vs_rig_{well_name}.pdf",
    )