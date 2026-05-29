from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from application.plotting_engine import plot_feasible_region_pretty, plot_contour_wraper
from configuration.wells import get_wells



# ============================================================
# Helpers
# ============================================================

def get_project_root():
    """
    Adjust if needed.

    If this script is inside:
        application/dissertation_figures/chapter4/

    then this returns:
        IOGR_StoAmaro/
    """
    return Path(__file__).resolve().parents[3]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sweep_for_well(well_name, filename="sweep_results.pkl"):
    project_root = get_project_root()
    sweep_path = project_root / "well_models" / well_name / filename

    if not sweep_path.exists():
        raise FileNotFoundError(f"Could not find sweep file: {sweep_path}")

    return load_pickle(sweep_path)


def mask_data(results, key):
    Z = np.asarray(results["OUT"][key], dtype=float).copy()
    return Z


# ============================================================
# Main plotting function
# ============================================================

def plot_contours_all_wells(
    wells,
    key,
    levels,
    sweep_filename="sweep_results.pkl",
    save=True,
    feasibility=False
):
    # --------------------------------------------------------
    # 1. Load all sweeps or run sweeps
    # --------------------------------------------------------
    results_all_wells = {
        well: load_sweep_for_well(well, filename=sweep_filename)
        for well in wells
    }

    # --------------------------------------------------------
    # 2. Create figure
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(7.2, 9.6),
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )

    axes = axes.ravel()

    cf = None

    # --------------------------------------------------------
    # 3. Plot each well
    # --------------------------------------------------------
    for ax, well in zip(axes, wells):
        results = results_all_wells[well]

        U1 = np.asarray(results["U1"], dtype=float)
        U2 = np.asarray(results["U2"], dtype=float)

        if feasibility:

            wells_dic=get_wells()

            key="P_bh_bar"
            bottomhole_pressure_values = mask_data(results, key)
            key = "P_tb_b_bar"
            tubing_pressure_values = mask_data(results, key)
            instability_coef_dict={"coef":wells_dic[well]["coeff_stability"]}

            fig, ax, feasible = plot_feasible_region_pretty(
                u1_grid=results["u1_grid"],
                u2_grid=results["u2_grid"],
                bottomhole_pressure_values=bottomhole_pressure_values,
                tubing_pressure_values=tubing_pressure_values,
                p_tb_max=120.0,
                p_bh_min=80.0,
                instability_coef_dict=instability_coef_dict,
                instability_side="above",
                n_fine=500,
                save_path="feasible_region.pdf",
                ax=ax
            )

        else:

            Z = mask_data(
                results,
                key=key
            )
            # --------------------------------------------------------
            # Minimum and maximum P_bh points
            # --------------------------------------------------------
            idx_min = np.nanargmin(Z)
            i_min, j_min = np.unravel_index(idx_min, Z.shape)

            u1_min_pbh = U1[i_min, j_min]
            u2_min_pbh = U2[i_min, j_min]
            z_min_pbh = Z[i_min, j_min]

            idx_max = np.nanargmax(Z)
            i_max, j_max = np.unravel_index(idx_max, Z.shape)

            u1_max_pbh = U1[i_max, j_max]
            u2_max_pbh = U2[i_max, j_max]
            z_max_pbh = Z[i_max, j_max]

            print(
                f"{well}: min {key} = {z_min_pbh:.4f} bar "
                f"at u1={u1_min_pbh:.4f}, u2={u2_min_pbh:.4f}"
            )

            print(
                f"{well}: max {key} = {z_max_pbh:.4f} bar "
                f"at u1={u1_max_pbh:.4f}, u2={u2_max_pbh:.4f}"
            )

            if key=="w_o_out":
                wells_dic=get_wells()
                u1_opt=wells_dic[well]["optima"]["surrogate"]["u"][0]
                u2_opt=wells_dic[well]["optima"]["surrogate"]["u"][1]
                w_o_out_opt=wells_dic[well]["optima"]["surrogate"]["w_o_out"]
                ax, cf = plot_contour_wraper(
                    results,
                    key=key,
                    title=f"{well}",
                    zlabel="P_bh",
                    only_stable=False,
                    only_success=False,
                    levels=levels,
                    mark_optimum=True,
                    u1_opt=u1_opt,
                    u2_opt=u2_opt,
                    z_opt=w_o_out_opt,
                    ax=ax
                )
            else:
                ax, cf = plot_contour_wraper(
                    results,
                    key=key,
                    title=f"{well}",
                    zlabel="P_bh",
                    only_stable=False,
                    only_success=False,
                    levels=levels,
                    mark_optimum=False,
                    ax=ax)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{well}",fontsize=14)
        ax.set_xlabel(r"$u_1$",fontsize=12)
        ax.set_ylabel(r"$u_2$",fontsize=12)

    # --------------------------------------------------------
    # 4. Shared colorbar
    # --------------------------------------------------------
    if not feasibility:
        fig.subplots_adjust(
            left=0.10,
            right=0.80,
            bottom=0.07,
            top=0.94,
            wspace=0.40,
            hspace=0.32,
        )

        cbar_ax = fig.add_axes([0.85, 0.10, 0.025, 0.75])

        cbar = fig.colorbar(
            cf,
            cax=cbar_ax,
        )
        if KEY=="P_bh_bar":
            cbar.set_label("p_bh (bar)")
        elif KEY=="w_G_inj":
            cbar.set_label("w_G_inj (kg/s)")
        elif KEY=="w_o_out":
            cbar.set_label("w_o_out (kg/s)")

    else:
        fig.subplots_adjust(
            left=0.10,
            right=0.90,
            bottom=0.07,
            top=0.94,
            wspace=0.40,
            hspace=0.40,
        )


    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------
    if save:
        project_root = get_project_root()
        save_dir = project_root / "application" / "dissertation_figures" / "chapter4"
        save_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = save_dir / f"{KEY}_contours_all_wells.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved PDF at: {pdf_path}")

    plt.show()

    return fig, axes, results_all_wells


if __name__ == "__main__":
    WELLS = ["P1", "P2", "P3", "P4", "P5", "P6"]
    KEY = "P_bh_bar"
    LEVELS = [
        72, 74,76, 78, 80, 82, 84, 86, 88, 90,
        92, 94, 96, 98, 100, 102, 104, 106, 108, 110,
        112, 114, 116, 118, 120, 122, 124, 126, 128, 130,
        132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152
    ]
    plot_contours_all_wells(
        wells=WELLS,
        key=KEY,
        levels=LEVELS,
        sweep_filename="sweep_results.pkl",
        save=True,
        feasibility=False
    )

    KEY = "w_G_inj"
    LEVELS = [
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
    ]
    plot_contours_all_wells(
        wells=WELLS,
        key=KEY,
        levels=LEVELS,
        sweep_filename="sweep_results_figures_training.pkl",
        save=True,
        feasibility=False
    )

    KEY = "w_o_out"
    LEVELS = [
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
    ]
    plot_contours_all_wells(
        wells=WELLS,
        key=KEY,
        levels=LEVELS,
        sweep_filename="sweep_results_figures_training.pkl",
        save=True,
        feasibility=False
    )
    KEY = "feasibility"
    plot_contours_all_wells(
        wells=WELLS,
        key=KEY,
        levels=None,
        sweep_filename="sweep_results_figures_training.pkl",
        save=True,
        feasibility=True
    )