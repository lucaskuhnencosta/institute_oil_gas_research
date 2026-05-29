from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configuration.wells import get_wells
from application.simulation_engine import make_glc_well_surrogate


# ============================================================
# User choices
# ============================================================

WELL_NAME = "P1"

# For PINN state variables, use one of:
# "m_G_an", "m_G_t", "m_o_t"

VARIABLE_NAME = "m_G_an"


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

def plot_sweep_surface_with_17_points(
    well_name,
    variable_name,
    elev=25,
    azim=-135,
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

    Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)

    U_17, Y_17 = extract_17_point_values(poly_dataset, variable_name)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        U1,
        U2,
        Y_sweep,
        linewidth=0,
        antialiased=True,
        alpha=0.75,
    )
    print(Y_sweep)
    ax.scatter(
        U_17[:, 0],
        U_17[:, 1],
        Y_17,
        marker="o",
        s=70,
        color="red",
        edgecolor="black",
        depthshade=True,
        label="17 selected points",
    )
    print(Y_17)

    # Optional: also plot red X markers on the bottom plane
    z_floor = np.nanmin(Y_sweep)

    ax.scatter(
        U_17[:, 0],
        U_17[:, 1],
        np.full_like(Y_17, z_floor),
        marker="x",
        s=80,
        color="red",
        label="17 point coordinates",
    )

    ax.set_title(f"{well_name} - {variable_name}: sweep surface and 17 selected points")
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")
    ax.set_zlabel(variable_name)

    ax.view_init(elev=elev, azim=azim)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "U1": U1,
        "U2": U2,
        "Y_sweep": Y_sweep,
        "U_17": U_17,
        "Y_17": Y_17,
    }


if __name__ == "__main__":
    data = plot_sweep_surface_with_17_points(
        well_name=WELL_NAME,
        variable_name=VARIABLE_NAME,
    )

