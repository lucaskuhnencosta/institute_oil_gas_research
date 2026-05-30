from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configuration.wells import get_wells
from application.simulation_engine import make_glc_well_surrogate


# ============================================================
# User choices
# ============================================================

WELL_NAME = "P5"

# For PINN state variables, use one of:
# "m_G_an", "m_G_t", "m_o_t"

VARIABLE_NAME = "P_bh_bar"


# ============================================================
# Helpers
# ============================================================

def get_project_root():
    """
    Assumes this script is executed from:
        application/model studies/
    so the project root is two folders above cwd.
    """
    return Path.cwd().parents[2]


def get_well_folder(well_name):
    return get_project_root() / "well_models" / str(well_name)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_17_point_values(poly_dataset, variable_name):
    """
    Extract the 17 selected values for the requested variable.

    The dataset contains:
        Y_pinn for states:
            ["m_G_an", "m_G_t", "m_o_t"]

        Y_poly for algebraic polynomial outputs:
            ["P_bh_bar", "P_tb_b_bar", "w_G_inj"]
    """

    if variable_name in poly_dataset["pinn_output_names"]:
        names = poly_dataset["pinn_output_names"]
        Y = np.asarray(poly_dataset["Y_pinn"], dtype=float)

    elif variable_name in poly_dataset["poly_output_names"]:
        names = poly_dataset["poly_output_names"]
        Y = np.asarray(poly_dataset["Y_poly"], dtype=float)

    else:
        raise ValueError(
            f"Variable {variable_name} not found. Available variables are:\n"
            f"PINN: {poly_dataset['pinn_output_names']}\n"
            f"POLY: {poly_dataset['poly_output_names']}"
        )

    k = names.index(variable_name)

    U_17 = np.asarray(poly_dataset["U"], dtype=float)
    Y_17 = Y[:, k]

    return U_17, Y_17


# ============================================================
# Main diagnostic plot
# ============================================================

def plot_sweep_surface_with_17_points_and_random_samples(
    well_name,
    variable_name,
    elev=15,
    azim=20,
    n_random=20,
    seed=123,
):
    folder = get_well_folder(well_name)

    sweep_path = folder / "sweep_results.pkl"
    dataset_path = folder / "poly_dataset.pkl"

    print(f"Loading sweep from:   {sweep_path}")
    print(f"Loading dataset from: {dataset_path}")

    sweep = load_pickle(sweep_path)
    poly_dataset = load_pickle(dataset_path)

    U1 = np.asarray(sweep["U1"], dtype=float)
    U2 = np.asarray(sweep["U2"], dtype=float)

    u1_min, u1_max = np.nanmin(U1), np.nanmax(U1)
    u2_min, u2_max = np.nanmin(U2), np.nanmax(U2)

    Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)

    U_17, Y_17 = extract_17_point_values(poly_dataset, variable_name)

    # =====================================================
    # Random samples in [u1_min, 1.0] x [u2_min, 1.0]
    # =====================================================
    rng = np.random.default_rng(seed)

    U_random = np.column_stack([
        rng.uniform(u1_min, 1.0, size=n_random),
        rng.uniform(u2_min, 1.0, size=n_random),
    ])

    # Put random samples on the bottom plane
    z_floor = np.nanmin(Y_sweep) - 5.0
    Z_random_floor = np.full(n_random, z_floor)

    # =====================================================
    # Figure
    # =====================================================
    fig = plt.figure(figsize=(7.2, 4.0))

    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    axes = [ax_left, ax_right]

    # Common z limits so both panels are visually comparable
    z_min = z_floor
    z_max = np.nanmax(Y_sweep)

    for ax in axes:
        ax.plot_surface(
            U1,
            U2,
            Y_sweep,
            linewidth=0,
            antialiased=True,
            alpha=0.70,
            cmap="viridis",
        )

        ax.set_xlabel("$u_1$")
        ax.set_ylabel("$u_2$")
        ax.set_zlabel(variable_name)

        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_zlim(z_min, z_max)

        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=elev, azim=azim)

    # =====================================================
    # Left: original 17 selected points
    # =====================================================
    ax_left.scatter(
        U_17[:, 0],
        U_17[:, 1],
        Y_17,
        marker="o",
        s=50,
        color="red",
        edgecolor="black",
        depthshade=True,
    )
    red_scatter=ax_right.scatter(
        U_17[:, 0],
        U_17[:, 1],
        Y_17,
        marker="o",
        s=50,
        color="red",
        edgecolor="black",
        depthshade=True,
    )



    # =====================================================
    # Right: 50 random points on bottom plane
    # =====================================================
    blue_scatter=ax_right.scatter(
        U_random[:, 0],
        U_random[:, 1],
        Z_random_floor,
        marker="x",
        s=50,
        color="blue",
        linewidths=1.8,
        depthshade=False,
    )


    ax_left.set_title(
        f"Polynomial interpolation", fontsize=14,pad=-10
    )
    ax_right.set_title(
        f"PINN-AlgNN Training", fontsize=14, pad=-10
    )

    fig.legend(
        handles=[red_scatter, blue_scatter],
        labels=["Simulated data points", "Collocation points"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=12,

        handletextpad=0.05,   # distance between marker and text
        columnspacing=1.5,   # distance between the two legend entries
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.11,
        top=0.98,
        wspace=0.12,
        hspace=0.10,
    )

    fig.savefig(
        "experiment_design.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()

    return {
        "U1": U1,
        "U2": U2,
        "Y_sweep": Y_sweep,
        "U_17": U_17,
        "Y_17": Y_17,
        "U_random": U_random,
        "z_floor": z_floor,
    }

# if __name__ == "__main__":
#     data = plot_sweep_surface_with_17_points(
#         well_name=WELL_NAME,
#         variable_name=VARIABLE_NAME,
#     )

if __name__ == "__main__":
    data = plot_sweep_surface_with_17_points_and_random_samples(
        well_name=WELL_NAME,
        variable_name=VARIABLE_NAME,
        n_random=40,
        seed=123,
    )