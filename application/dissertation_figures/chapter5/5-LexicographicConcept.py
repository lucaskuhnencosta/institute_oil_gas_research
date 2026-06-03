from application.producation_optim_run import production_optim_run
import pandas as pd
from configuration.wells import get_wells
import numpy as np
from application.plotting_engine import plot_feasible_region_pretty
from settings import U1_MIN
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

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

def obtain_df_views():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    u_guess=np.array([[U1_MIN,1.0]])
    cols = [
        "phase",
        "iteration",
        "u",
        "u_trial",
        "Delta",
        "theta_k",
        "phi_k",
        "type"
    ]
    result1, result2 = production_optim_run(
        selected_wells=["P1"],
        u_guess=u_guess
    )
    df_history_1 = pd.DataFrame(result1["history"])
    df_view_1 = df_history_1[cols].copy()
    df_history_2=pd.DataFrame(result2["history"])
    df_view_2 = df_history_2[cols].copy()


    # 1) Pile them up
    df_plot = pd.concat([df_view_1, df_view_2], ignore_index=True)

    # 2) Filter out unwanted step types
    steps_to_remove = [
        "no-step-converged",
        "init",
        "convergence"
    ]

    df_plot = df_plot[~df_plot["type"].isin(steps_to_remove)].copy()

    # 3) Substitute remaining step types by colors
    step_color_map = {
        "restoration": "blue",
        "oil_maximization": "blue",
        "refinement": "purple"
    }

    df_plot["phase"] = df_plot["phase"].replace(step_color_map)

    return df_plot

if __name__ == "__main__":
    sweep_filename = "sweep_results_figures_training.pkl"
    wells = get_wells()

    results_all_wells = {
        well: load_sweep_for_well(well, filename=sweep_filename)
        for well in wells
    }
    well_name="P1"
    well=wells[well_name]
    results = results_all_wells[well_name]

    U1 = np.asarray(results["U1"], dtype=float)
    U2 = np.asarray(results["U2"], dtype=float)


    key = "P_bh_bar"
    bottomhole_pressure_values = mask_data(results, key)
    key = "P_tb_b_bar"
    tubing_pressure_values = mask_data(results, key)
    key="w_o_out"
    w_o_out_vales = mask_data(results, key)
    instability_coef_dict = {"coef": wells[well_name]["coeff_stability"]}

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 4.2),
        sharex=True,
        sharey=True
    )



    ax_left, ax_right = axes

    _, ax_left, legend_elements = plot_feasible_region_pretty(
        u1_grid=results["u1_grid"],
        u2_grid=results["u2_grid"],
        bottomhole_pressure_values=bottomhole_pressure_values,
        tubing_pressure_values=tubing_pressure_values,
        w_o_out_values=w_o_out_vales,
        p_tb_max=120.0,
        p_bh_min=80.0,
        instability_coef_dict=instability_coef_dict,
        instability_side="above",
        n_fine=500,
        save_path=None,
        ax=ax_left
    )


    _, ax_right, legend_elements = plot_feasible_region_pretty(
        u1_grid=results["u1_grid"],
        u2_grid=results["u2_grid"],
        bottomhole_pressure_values=bottomhole_pressure_values,
        tubing_pressure_values=tubing_pressure_values,
        w_o_out_values=w_o_out_vales,
        p_tb_max=120.0,
        p_bh_min=80.0,
        instability_coef_dict=instability_coef_dict,
        instability_side="above",
        n_fine=500,
        save_path=None,
        phase_two_feasibility=True,
        w_o_out_phase_2=21.596,
        ax=ax_right
    )

    ax_left.set_title("Primary objective")
    ax_right.set_title("Secondary objective")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")

    df_plot=obtain_df_views()
    print(df_plot)
    df_plot = df_plot.drop(index=[12,13])
    print(df_plot)
    for _, row in df_plot.iterrows():
        k=row["iteration"]
        u = np.array(row["u"], dtype=float)
        u_trial = np.array(row["u_trial"], dtype=float)
        radius = float(row["Delta"])

        phase = row["phase"]

        if phase == "blue":
            ax = ax_left
            color = "blue"
        elif phase == "purple":
            ax = ax_right
            color = "purple"
        else:
            continue

        # Line from u to u_trial
        ax.plot(
            [u[0], u_trial[0]],
            [u[1], u_trial[1]],
            linestyle="--",
            color=color,
            linewidth=1.5,
            zorder=10
        )

        # Points u and u_trial
        ax.scatter(
            [u[0], u_trial[0]],
            [u[1], u_trial[1]],
            color=color,
            s=35,
            zorder=11
        )

        if not(row["type"] =="accepted restoration" or (row["phase"]=="blue" and row["iteration"]==3) or (row["phase"]=="purple" and row["iteration"]==2)):
            # Circle centered at u
            circle_u = Circle(
                (u[0], u[1]),
                radius=radius,
                fill=False,
                edgecolor="grey",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=9
            )
            ax.add_patch(circle_u)


            # Circle number label, placed near top-right of circle
            if k==0:
                ax.text(
                    u[0] + 0.85 * radius,
                    u[1] + 0.35 * radius,
                    f"${k}$",
                    color=color,
                    fontsize=11,
                    ha="center",
                    va="center",
                    zorder=13
                )

            else:
                ax.text(
                    u[0] - 0.65* radius,
                    u[1] - 0.80 * radius,
                    f"{k}",
                    color=color,
                    fontsize=11,
                    ha="center",
                    va="center",
                    zorder=13
                )
            # Point label u_k
            if phase == "blue":
                ax.text(
                    u[0] + 0.010,
                    u[1] + 0.00025,
                    rf"$\mathbf{{u}}_{{{k}}}$",
                    color=color,
                    fontsize=11,
                    ha="left",
                    va="bottom",
                    zorder=13
                )
            else:
                ax.text(
                    u[0] - 0.075,
                    u[1] + 0.00025,
                    rf"$\mathbf{{u}}_{{{k}}}$",
                    color=color,
                    fontsize=11,
                    ha="left",
                    va="bottom",
                    zorder=13
                )
            if (row["phase"]=="blue" and row["iteration"]==2) or (row["phase"]=="purple" and row["iteration"]==1):
                if row["phase"] == "blue":
                    ax.text(
                        u_trial[0] - 0.07,
                        u_trial[1] - 0.07,
                        rf"$\mathbf{{u}}_{{{3}}}$",
                        color=color,
                        fontsize=11,
                        ha="left",
                        va="bottom",
                        zorder=13
                    )
                else:
                    ax.text(
                        u_trial[0] + 0.015,
                        u_trial[1] + 0.00025,
                        rf"$\mathbf{{u}}_{{{2}}}$",
                        color=color,
                        fontsize=11,
                        ha="left",
                        va="bottom",
                        zorder=13
                    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=11,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.20,
        top=0.92,
        wspace=0.20
    )

    fig.savefig("Lexicographic_Optimization.pdf",
                bbox_inches="tight")

    plt.show()


