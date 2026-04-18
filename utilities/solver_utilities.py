import numpy as np
from collections import defaultdict


def print_z_grouped(z_star, names, floatfmt="{: .6f}"):
    z = np.array(z_star).astype(float).reshape(-1)

    if len(z) != len(names):
        raise ValueError(
            f"z length mismatch: got {len(z)} values but {len(names)} names"
        )

    pairs = list(zip(names, z))

    # -----------------------------
    # Define grouping rules
    # -----------------------------
    groups = defaultdict(list)

    for name, val in pairs:

        if name.startswith("P_"):
            groups["Pressures"].append((name, val))

        elif name.startswith("dP_"):
            groups["Pressure Drops"].append((name, val))

        elif name.startswith("F_"):
            groups["Friction Terms"].append((name, val))

        elif name.startswith("w_"):
            groups["Mass Flows"].append((name, val))

        elif name.startswith("rho_"):
            groups["Densities"].append((name, val))

        elif name.startswith("U_"):
            groups["Velocities"].append((name, val))

        elif name.startswith("Re_"):
            groups["Reynolds Numbers"].append((name, val))

        elif name.startswith("alpha_"):
            groups["Alphas (Fractions)"].append((name, val))

        elif name.startswith("V_"):
            groups["Volumes"].append((name, val))

        elif name.startswith("Q_"):
            groups["Volumetric Flows"].append((name, val))

        elif name.startswith("lambda_"):
            groups["Friction Factors"].append((name, val))

        else:
            groups["Other"].append((name, val))

    # -----------------------------
    # Print neatly
    # -----------------------------
    for group_name in sorted(groups.keys()):
        print("\n" + "="*60)
        print(f"{group_name.upper()}")
        print("="*60)

        for name, val in groups[group_name]:
            print(f"{name:25s} = {floatfmt.format(val)}")