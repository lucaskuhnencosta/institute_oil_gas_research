from application.producation_optim_run import production_optim_run
import numpy as np
import ast
import pandas as pd
from application.plotting_engine import plot_feasible_region_pretty
from pathlib import Path
import pickle
from configuration.wells import get_wells
import numpy as np
import ast
import matplotlib.pyplot as plt

wells = get_wells()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_well_folder(well_name):
    return Path.cwd().parents[2] / "well_models" / str(well_name)

def to_array(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    return np.asarray(x, dtype=float).reshape(-1)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

u_guess_1=np.array([[1.00,1.00]])
u_guess_2=np.array([[1.00,0.15]])
u_guess_3=np.array([[0.10,1.00]])

u_guess_list=[u_guess_1,u_guess_2,u_guess_3]
cols = [
    "phase",
    "iteration",
    "u",
    "u_trial",
    "Delta",
    "accepted",
    "rejected",
    "theta_k",
]
all_runs = []

for run_id, u_guess in enumerate(u_guess_list):
    result = production_optim_run(
        u_guess=u_guess,
        config=None,
        selected_wells=["P2"],
    )

    df_history = pd.DataFrame(result["history"])
    df_view = df_history[cols].copy()

    df_view["run_id"] = run_id
    df_view["u_guess"] = [u_guess] * len(df_view)

    all_runs.append(df_view)

df_all_runs = pd.concat(all_runs, ignore_index=True)
print(df_all_runs.to_string(index=False))

well_name="P2"
folder = get_well_folder(well_name)
sweep_path = folder / "sweep_results_figures_training.pkl"
sweep = load_pickle(sweep_path)
u1_grid = np.asarray(sweep["u1_grid"], dtype=float)
u2_grid = np.asarray(sweep["u2_grid"], dtype=float)
U1 = sweep["U1"]
U2 = sweep["U2"]
instability_coef_dict = {"coef": wells[well_name]["coeff_stability"]}

fig,(ax_region,ax_theta)=plt.subplots(1,2,figsize=(7.2,3.5))


fig, ax_region, feasible = plot_feasible_region_pretty(
    u1_grid=u1_grid,
    u2_grid=u2_grid,
    bottomhole_pressure_values=sweep["OUT"]["P_bh_bar"],
    tubing_pressure_values=sweep["OUT"]["P_tb_b_bar"],
    p_tb_max=120.0,
    p_bh_min=80.0,
    instability_coef_dict=instability_coef_dict ,
    instability_side="above",
    n_fine=500,
    ax=ax_region)


for run_id, df_run in df_all_runs.groupby("run_id"):
    df_run = df_run.sort_values("iteration")

    u_values = np.vstack(df_run["u"].apply(to_array).values)

    u1 = u_values[:, 0]
    u2 = u_values[:, 1]

    if run_id == 0:
        color="blue"
    elif run_id == 1:
        color="red"
    elif run_id == 2:
        color="purple"
    else:
        color="darkorange"

    line, = ax_region.plot(
        u1,
        u2,
        "--",
        color=color,
        linewidth=1.5,
        label=rf"$\mathbf{{u}}_0 = ({u1[0]:.2f}, {u2[0]:.2f})$",
    )

    ax_region.scatter(
        u1,
        u2,
        s=35,
        color=color,
        zorder=5,
    )


    ax_theta.plot(
        df_run["iteration"],
        df_run["theta_k"],
        "--o",
        linewidth=1.5,
        markersize=4,
        color=color,
        label=f"Run {run_id}",
    )



ax_region.set_title("Phase 1 trajectories")
ax_theta.set_title(r"Constraint violation along iterations")


from matplotlib.patches import Patch

ax_theta.set_xlabel("Iteration",fontsize=12)
ax_theta.set_ylabel(r"$v_{k}$",fontsize=12)
ax_theta.grid(False)

handles, labels = ax_region.get_legend_handles_labels()

feasible_handle = Patch(
    facecolor="green",
    edgecolor="none",
    alpha=0.20,
    label="Feasible region",
)

handles = [feasible_handle] + handles
labels = ["Feasible region"] + labels

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, -0.015),
    columnspacing=0.7,
    fontsize=11,
    handletextpad=0.2,
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.22,
    top=0.90,
    wspace=0.30,
)

fig.savefig(
    "phase1_trajectories_and_theta.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()



