import numpy as np
from pathlib import Path
import pickle
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from application.simulation_engine import run_sweep
from application.simulation_engine import make_model

from configuration.wells import get_wells

from pathlib import Path
import pickle
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from pathlib import Path
import pickle
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from settings import U1_MIN

####################################################################################################
# THIS PART HERE HAS ALREADY BEEN USED AND WILL NOT BE USED AGAIN
####################################################################################################

def poly_dataset(
    sweep_result,
    well_name,
    poly_output_names=("P_bh_bar", "P_tb_b_bar", "w_G_inj"),
    pinn_output_names=("m_G_an","m_G_t","m_o_t"),
    save=True,
):
    U_scale = np.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],

        [-1.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],

        [-0.5,  0.0],
        [ 0.5,  0.0],
        [ 0.0, -0.5],
        [ 0.0,  0.5],

        [-0.5, -0.5],
        [-0.5,  0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],

        [ 0.0,  0.0],
    ], dtype=float)

    OUT = sweep_result["OUT"]
    u1_grid = np.asarray(sweep_result["u1_grid"], dtype=float)
    u2_grid = np.asarray(sweep_result["u2_grid"], dtype=float)

    u1_min = float(u1_grid[0])
    u1_max = float(u1_grid[-1])
    u2_min = float(u2_grid[0])
    u2_max = float(u2_grid[-1])

    # Convert hardcoded scaled points to physical input values
    U_target = np.zeros_like(U_scale)
    U_target[:, 0] = u1_min + 0.5 * (U_scale[:, 0] + 1.0) * (u1_max - u1_min)
    U_target[:, 1] = u2_min + 0.5 * (U_scale[:, 1] + 1.0) * (u2_max - u2_min)

    U_grid = []
    Y_poly = []
    Y_pinn = []
    grid_indices = []

    for u1_target, u2_target in U_target:
        i = int(np.argmin(np.abs(u1_grid - u1_target)))
        j = int(np.argmin(np.abs(u2_grid - u2_target)))

        U_grid.append([u1_grid[i], u2_grid[j]])
        grid_indices.append([i, j])

        Y_poly.append([
            OUT[name][i, j]
            for name in poly_output_names
        ])

        Y_pinn.append([
            OUT[name][i,j]
            for name in pinn_output_names
        ])

    U_grid = np.asarray(U_grid, dtype=float)
    Y_poly = np.asarray(Y_poly, dtype=float)
    Y_pinn = np.asarray(Y_pinn, dtype=float)
    grid_indices = np.asarray(grid_indices, dtype=int)

    # Recompute the actual scaled coordinates of the selected grid points.
    # This matters because the 20-by-20 grid may not contain exactly -0.5, 0, 0.5.
    U_scale_grid = np.zeros_like(U_grid)
    U_scale_grid[:, 0] = 2.0 * (U_grid[:, 0] - u1_min) / (u1_max - u1_min) - 1.0
    U_scale_grid[:, 1] = 2.0 * (U_grid[:, 1] - u2_min) / (u2_max - u2_min) - 1.0

    dataset = {
        "well_name": well_name,
        "input_names": ["u1", "u2"],

        "poly_output_names": list(poly_output_names),
        "pinn_output_names": list(pinn_output_names),

        "U_scale_design": U_scale,
        "U_target": U_target,

        # These are the actual grid values used for fitting
        "U": U_grid,
        "U_scale": U_scale_grid,

        "Y_poly": Y_poly,
        "Y_pinn": Y_pinn,

        "output_names": list(poly_output_names),
        "Y": Y_poly,

        "grid_indices": grid_indices,

        "input_scaling": {
            "u1_min": u1_min,
            "u1_max": u1_max,
            "u2_min": u2_min,
            "u2_max": u2_max,
        },
    }

    if save:
        folder = Path.cwd().parent / "well_models" / str(well_name)
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / "poly_dataset.pkl"
        with open(path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Saved dataset: {path}")

    return dataset


def fit_cubic_polynomial(dataset, save=True):
    """
    Fit vector-valued cubic polynomial:

        Y = polynomial(U_scale)

    where U_scale is the actual selected grid point in scaled coordinates.
    """

    well_name = dataset["well_name"]

    U_scale = np.asarray(dataset["U_scale"], dtype=float)
    Y = np.asarray(dataset["Y_poly"], dtype=float)
    output_names = dataset["poly_output_names"]

    poly = PolynomialFeatures(degree=3, include_bias=True)
    Phi = poly.fit_transform(U_scale)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(Phi, Y)

    Y_hat = reg.predict(Phi)

    coefficients = reg.coef_.T
    powers = poly.powers_

    rmse = np.sqrt(np.mean((Y - Y_hat) ** 2, axis=0))
    max_abs_error = np.max(np.abs(Y - Y_hat), axis=0)

    r2 = np.array([
        r2_score(Y[:, k], Y_hat[:, k])
        for k in range(Y.shape[1])
    ])

    model_dict = {
        "well_name": well_name,
        "model_type": "cubic_polynomial",
        "degree": 3,

        "input_names": dataset["input_names"],
        "output_names": list(output_names),

        "input_scaling": dataset["input_scaling"],

        "powers": powers.tolist(),
        "coefficients": coefficients.tolist(),

        "metrics": {
            "rmse": {
                name: float(rmse[k])
                for k, name in enumerate(dataset["output_names"])
            },
            "r2": {
                name: float(r2[k])
                for k, name in enumerate(dataset["output_names"])
            },
            "max_abs_error": {
                name: float(max_abs_error[k])
                for k, name in enumerate(dataset["output_names"])
            },
        },

        "training_data": {
            "U": dataset["U"].tolist(),
            "U_scale": U_scale.tolist(),
            "Y": Y.tolist(),
            "Y_hat": Y_hat.tolist(),
            "grid_indices": dataset["grid_indices"].tolist(),
        },
    }

    if save:
        folder = Path.cwd().parent / "well_models" / str(well_name)
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / "cubic_polynomial_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)

        print(f"Saved cubic model: {path}")

    print(f"\nCubic polynomial fit: {well_name}")
    print("-" * 60)

    for name in dataset["output_names"]:
        print(
            f"{name:15s} | "
            f"RMSE = {model_dict['metrics']['rmse'][name]:.6e} | "
            f"R2 = {model_dict['metrics']['r2'][name]:.8f} | "
            f"MaxAbs = {model_dict['metrics']['max_abs_error'][name]:.6e}"
        )

    return model_dict


def build_and_run_surrogate_sweep(
    u1_min: float,
    u2_min: float,
    N_data: int,
    y_guess_init,
    BSW: float,
    GOR: float,
    PI: float,
    K_gs: float,
    K_inj: float,
    K_pr: float,
    well_name: str,
    save: bool,
):
    """
    Creates the surrogate model, runs run_sweep on an N_data x N_data grid and get the results
    """

    model_sur = make_model("surrogate",
                           BSW=BSW,
                           GOR=GOR,
                           PI=PI,
                           K_gs=K_gs,
                           K_inj=K_inj,
                           K_pr=K_pr)

    results = run_sweep(
        model_sur,
        U1_MIN=u1_min,
        U2_MIN=u2_min,
        U_SIM_SIZE=N_data,
        y_guess_init=y_guess_init)

    if save:
        if well_name is None:
            raise ValueError("well_name must be provided when save=True.")

        folder = Path.cwd().parent / "well_models" / str(well_name)
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / "sweep_results.pkl"

        with open(path, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved sweep results: {path}")

    return results

####################################################################################################
# END OF THE PART USED
# ##################################################################################################


if __name__ == "__main__":
    import numpy as np
    from configuration.wells import get_wells
    from settings import *
    N_data = 20
    wells = get_wells()
    for well in wells:

        well_name=well
        BSW = wells[well]["BSW"]
        GOR = wells[well]["GOR"]
        PI = wells[well]["PI"]
        K_gs = wells[well]["K_gs"]
        K_inj = wells[well]["K_inj"]
        K_pr = wells[well]["K_pr"]
        y_guess_sur = wells[well]["y_guess_sur"]

        u1_min, u1_max = U1_MIN, U1_MAX
        u2_min, u2_max = U2_MIN, U2_MAX



        y_guess_sur = np.array(y_guess_sur, dtype=float)

        print("Running sweep...")
        results = build_and_run_surrogate_sweep(
            u1_min=u1_min,
            u2_min=u2_min,
            N_data=N_data,
            y_guess_init=y_guess_sur,
            BSW=BSW,
            GOR=GOR,
            PI=PI,
            K_gs=K_gs,
            K_inj=K_inj,
            K_pr=K_pr,
            well_name=well_name,
            save=True
        )

def flatten_sweep_results_to_batch_full(
    results: dict,
    only_success: bool = True,
    y_names=None,
    z_names=None,
):
    """
    Flatten a saved sweep dictionary into tensors for AlgNN training.

    AlgNN dataset:
        input  -> (y, u)
        target -> z

    where:
        y = selected state variables
        u = [u1, u2]
        z = selected algebraic/output variables
    """

    import numpy as np
    import torch

    if y_names is None:
        y_names = ["m_G_an", "m_G_t", "m_o_t"]

    if z_names is None:
        z_names = ["P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res"]

    u1_grid = np.asarray(results["u1_grid"], dtype=float)
    u2_grid = np.asarray(results["u2_grid"], dtype=float)

    OUT = results["OUT"]

    Nu1 = len(u1_grid)
    Nu2 = len(u2_grid)

    # ------------------------------------------------------------
    # Flatten control grid
    # ------------------------------------------------------------
    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")

    u_flat = np.stack(
        [U1.reshape(-1), U2.reshape(-1)],
        axis=1,
    )

    # ------------------------------------------------------------
    # Flatten state inputs y
    # ------------------------------------------------------------
    y_flat = np.stack(
        [
            np.asarray(OUT[name], dtype=float).reshape(-1)
            for name in y_names
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # Flatten algebraic targets z
    # ------------------------------------------------------------
    z_flat = np.stack(
        [
            np.asarray(OUT[name], dtype=float).reshape(-1)
            for name in z_names
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # Build mask
    # ------------------------------------------------------------
    finite_u = np.all(np.isfinite(u_flat), axis=1)
    finite_y = np.all(np.isfinite(y_flat), axis=1)
    finite_z = np.all(np.isfinite(z_flat), axis=1)

    finite_all = finite_u & finite_y & finite_z

    if only_success and "SUCCESS" in results:
        success_flat = np.asarray(results["SUCCESS"], dtype=bool).reshape(-1)
        mask = success_flat & finite_all
    else:
        mask = finite_all

    # ------------------------------------------------------------
    # Apply mask
    # ------------------------------------------------------------
    u_np = u_flat[mask].astype(np.float32)
    y_np = y_flat[mask].astype(np.float32)
    z_np = z_flat[mask].astype(np.float32)

    # ------------------------------------------------------------
    # Torch tensors
    # ------------------------------------------------------------
    u_t = torch.tensor(u_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    z_t = torch.tensor(z_np, dtype=torch.float32)

    return {
        "y_names": list(y_names),
        "z_names": list(z_names),

        "u_np": u_np,
        "y_np": y_np,
        "z_np": z_np,

        "u_t": u_t,
        "y_t": y_t,
        "z_t": z_t,

        "Nu1": Nu1,
        "Nu2": Nu2,
        "mask_np": mask,
    }


