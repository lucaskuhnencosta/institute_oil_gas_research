from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configuration.wells import get_wells
from application.simulation_engine import make_glc_well_surrogate

# ============================================================
# Helpers
# ============================================================

def get_project_root():
    return Path.cwd().parents[2]


def get_well_folder(well_name):
    return get_project_root() / "well_models" / str(well_name)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User choices
# ============================================================

WELL_NAME = "P5"

N_RANDOM =1000
SEED = 123

U1_MIN = 0.05
U1_MAX = 1.00

U2_MIN = 0.10
U2_MAX = 1.00

SAVE_FIGURE = True
FIGURE_NAME = "experiment_design_2d.pdf"


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


# ============================================================
# Main plot
# ============================================================

def plot_experiment_design_2d(
    well_name,
    n_random=100,
    seed=123,
    save_figure=True,
    figure_name="experiment_design_2d.pdf",
):
    folder = get_well_folder(well_name)
    dataset_path = folder / "poly_dataset.pkl"

    print(f"Loading dataset from: {dataset_path}")

    poly_dataset = load_pickle(dataset_path)

    # --------------------------------------------------------
    # 17 simulated data points
    # --------------------------------------------------------
    U_17 = np.asarray(poly_dataset["U"], dtype=float)

    # --------------------------------------------------------
    # Random collocation points
    # --------------------------------------------------------
    # rng = np.random.default_rng(seed)
    #
    # U_random = np.column_stack([
    #     rng.uniform(U1_MIN, U1_MAX, size=n_random),
    #     rng.uniform(U2_MIN, U2_MAX, size=n_random),
    # ])
    U_random = latin_hypercube_sampling(
        n_samples=n_random,
        bounds=[
            (U1_MIN, U1_MAX),
            (U2_MIN, U2_MAX),
        ],
        seed=seed,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.6),
        sharex=True,
        sharey=True,
    )

    ax_left, ax_right = axes

    # ========================================================
    # Left panel: polynomial interpolation
    # ========================================================
    ax_left.scatter(
        U_17[:, 0],
        U_17[:, 1],
        marker="o",
        s=60,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label="Simulated data points",
        zorder=3,
    )

    ax_left.set_title("Polynomial interpolation", fontsize=13)

    # ========================================================
    # Right panel: PINN-AlgNN training
    # ========================================================
    red_scatter = ax_right.scatter(
        U_17[:, 0],
        U_17[:, 1],
        marker="o",
        s=60,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label="Simulated data points",
        zorder=3,
    )

    blue_scatter = ax_right.scatter(
        U_random[:, 0],
        U_random[:, 1],
        marker="x",
        s=55,
        color="blue",
        linewidths=1.8,
        label="Collocation points",
        zorder=2,
    )

    ax_right.set_title("PINN-AlgNN training", fontsize=13)

    # ========================================================
    # Common formatting
    # ========================================================
    for ax in axes:
        ax.set_xlim(U1_MIN-0.05, U1_MAX+0.05)
        ax.set_ylim(U2_MIN-0.05, U2_MAX+0.05)

        ax.set_xlabel("$u_1$", fontsize=13)
        ax.set_aspect("equal", adjustable="box")

        # ax.grid(
        #     True,
        #     linestyle="--",
        #     linewidth=0.6,
        #     alpha=0.5,
        # )

    ax_left.set_ylabel("$u_2$", fontsize=13)

    # Shared legend
    fig.legend(
        handles=[red_scatter, blue_scatter],
        labels=["Simulated data points", "Collocation points"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=11,
        handletextpad=0.3,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.20,
        top=0.88,
        wspace=0.12,
    )

    if save_figure:
        fig.savefig(
            figure_name,
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

    return {
        "U_17": U_17,
        "U_random": U_random,
    }

def latin_hypercube_sampling(n_samples, bounds, seed=123):
    """
    Latin Hypercube Sampling in a rectangular domain.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of tuple
        Variable bounds, e.g. [(u1_min, u1_max), (u2_min, u2_max)].
    seed : int
        Random seed.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dimensions)
        LHS samples scaled to the requested bounds.
    """
    rng = np.random.default_rng(seed)

    n_dim = len(bounds)
    samples_unit = np.zeros((n_samples, n_dim))

    for j in range(n_dim):
        # One sample inside each interval
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        points = cut[:-1] + rng.random(n_samples) / n_samples

        # Randomly permute intervals for this dimension
        rng.shuffle(points)

        samples_unit[:, j] = points

    samples = np.zeros_like(samples_unit)

    for j, (lower, upper) in enumerate(bounds):
        samples[:, j] = lower + samples_unit[:, j] * (upper - lower)

    return samples

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    data = plot_experiment_design_2d(
        well_name=WELL_NAME,
        n_random=N_RANDOM,
        seed=SEED,
        save_figure=SAVE_FIGURE,
        figure_name=FIGURE_NAME,
    )
