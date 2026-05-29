from pathlib import Path
import pickle
import numpy as np


# ============================================================
# User settings
# ============================================================

STATE_NAMES = ["m_G_an", "m_G_t", "m_o_t"]

# Add a safety margin so the bounded PINN output can represent
# all sweep values without saturating exactly at the boundary.
MARGIN_FRAC = 0.05


# ============================================================
# Helpers
# ============================================================

def get_project_root():
    """
    Adjust if needed.

    If this script is executed from:
        application/dissertation_figures/chapter4/
    then project root is three folders above cwd.
    """
    return Path.cwd().parent


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def discover_y_ranges_for_well(well_folder, state_names=STATE_NAMES, margin_frac=MARGIN_FRAC):
    sweep_path = well_folder / "sweep_results.pkl"

    if not sweep_path.exists():
        print(f"Skipping {well_folder.name}: no sweep_results.pkl")
        return None

    sweep = load_pickle(sweep_path)

    ranges = {
        "well_name": well_folder.name,
        "raw_min": {},
        "raw_max": {},
        "margin_min": {},
        "margin_max": {},
    }

    for name in state_names:
        if name not in sweep["OUT"]:
            raise KeyError(
                f"{name} not found in {sweep_path}. "
                f"Available keys: {list(sweep['OUT'].keys())}"
            )

        values = np.asarray(sweep["OUT"][name], dtype=float)

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

        span = vmax - vmin
        margin = margin_frac * span

        ranges["raw_min"][name] = vmin
        ranges["raw_max"][name] = vmax
        ranges["margin_min"][name] = vmin - margin
        ranges["margin_max"][name] = vmax + margin

    return ranges


def discover_all_y_ranges():
    project_root = get_project_root()
    well_models_dir = project_root / "well_models"

    print(f"Project root: {project_root}")
    print(f"Well models dir: {well_models_dir}")

    all_ranges = {}

    for well_folder in sorted(well_models_dir.iterdir()):
        if not well_folder.is_dir():
            continue

        result = discover_y_ranges_for_well(well_folder)

        if result is None:
            continue

        well_name = result["well_name"]
        all_ranges[well_name] = result

    return all_ranges


def print_ranges_for_wells(all_ranges):
    print("\n" + "=" * 100)
    print("Discovered PINN state ranges from full sweep")
    print("=" * 100)

    for well_name, r in all_ranges.items():
        raw_min = [r["raw_min"][name] for name in STATE_NAMES]
        raw_max = [r["raw_max"][name] for name in STATE_NAMES]

        margin_min = [r["margin_min"][name] for name in STATE_NAMES]
        margin_max = [r["margin_max"][name] for name in STATE_NAMES]

        print(f"\n{well_name}")
        print("-" * 80)
        print(f"raw y_min    = {raw_min}")
        print(f"raw y_max    = {raw_max}")
        print(f"margin y_min = {margin_min}")
        print(f"margin y_max = {margin_max}")

        print("\nPaste into wells.py:")
        print(f'"y_min": {margin_min},')
        print(f'"y_max": {margin_max},')


def save_ranges(all_ranges):
    project_root = get_project_root()
    save_path = project_root / "well_models" / "discovered_y_ranges.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(all_ranges, f)

    print(f"\nSaved discovered ranges to: {save_path}")


if __name__ == "__main__":
    all_ranges = discover_all_y_ranges()
    print_ranges_for_wells(all_ranges)
    save_ranges(all_ranges)