import torch
import casadi as ca
import pandas as pd
from configuration.wells import get_wells
import numpy as np

from simulators.surrogate_simulator.surrogate_model_torch import glc_surrogate_dx_torch
from simulators.surrogate_simulator.surrogate_model_casadi import make_glc_well_surrogate
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configuration.wells import get_wells
from application.simulation_engine import make_glc_well_surrogate

CASADI_SMALL_Z_INDICES = {
    "P_bh_bar": 17,
    "P_tb_b_bar": 19,
    "w_G_inj": 41,
}

Z_NAMES_SMALL = [
    "P_bh_bar",
    "P_tb_b_bar",
    "w_G_inj",
]

def get_project_root():
    """
    Assumes this script is executed from:
        application/model studies/
    so the project root is two folders above cwd.
    """
    return Path.cwd().parents[2]

def get_well_folder(well_name):
    return get_project_root() / "well_models" / str(well_name)

def get_17_state_data(poly_dataset):
    """
    Extract the 17 input points and the 17 PINN state targets.

    Expected state order:
        [m_G_an, m_G_t, m_o_t]
    """

    U_17 = np.asarray(poly_dataset["U"], dtype=float)
    Y_17 = np.asarray(poly_dataset["Y_pinn"], dtype=float)

    state_names = poly_dataset["pinn_output_names"]

    return U_17, Y_17, state_names


def evaluate_dx_casadi_17_points(well_name,
                                 U_17,
                                 Y_17,
                                 BSW,
                                 GOR,
                                 PI,
                                 K_gs,
                                 K_inj,
                                 K_pr):
    """
    Evaluate CasADi residual dx at the 17 selected points.
    """

    glc = make_glc_well_surrogate(
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr,
    )

    dx_list = []
    z_list=[]

    idx = [
        CASADI_SMALL_Z_INDICES["P_bh_bar"],
        CASADI_SMALL_Z_INDICES["P_tb_b_bar"],
        CASADI_SMALL_Z_INDICES["w_G_inj"],
    ]

    for k in range(U_17.shape[0]):
        y_k = ca.DM(Y_17[k, :])
        u_k = ca.DM(U_17[k, :])

        dx_k, z_k = glc(y_k, u_k)

        dx_k = np.asarray(dx_k, dtype=float).reshape(-1)
        z_k = np.asarray(z_k, dtype=float).reshape(-1)

        dx_list.append(dx_k)
        z_list.append(z_k[idx])

    return np.asarray(dx_list), np.asarray(z_list)


def evaluate_dx_torch_17_points(well_name,
                                U_17,
                                Y_17,
                                BSW,
                                GOR,
                                PI,
                                K_gs,
                                K_inj,
                                K_pr):
    """
    Evaluate Torch residual dx at the 17 selected points.
    """

    device = torch.device("cpu")

    u_t = torch.tensor(U_17, dtype=torch.float64, device=device)
    y_t = torch.tensor(Y_17, dtype=torch.float64, device=device)

    with torch.no_grad():
        dx_t,z_t = glc_surrogate_dx_torch(
            y=y_t,
            u=u_t,
            BSW=BSW,
            GOR=GOR,
            PI=PI,
            K_gs=K_gs,
            K_inj=K_inj,
            K_pr=K_pr,
            return_z=True
        )

    return dx_t.detach().cpu().numpy(), z_t.detach().cpu().numpy()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compare_small_z_on_17_points(well_name):
    folder = get_well_folder(well_name)
    dataset_path = folder / "poly_dataset.pkl"

    poly_dataset = load_pickle(dataset_path)

    U_17 = np.asarray(poly_dataset["U"], dtype=float)
    Y_17 = np.asarray(poly_dataset["Y_pinn"], dtype=float)

    dx_ca, z_ca = evaluate_dx_casadi_17_points(
        well_name=well_name,
        U_17=U_17,
        Y_17=Y_17,
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr
    )

    dx_th, z_th = evaluate_dx_torch_17_points(
        well_name=well_name,
        U_17=U_17,
        Y_17=Y_17,
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr,
    )

    dx_diff = dx_th - dx_ca
    z_diff = z_th - z_ca

    print("\n" + "=" * 100)
    print(f"CasADi vs Torch on 17 points for well {well_name}")
    print("=" * 100)

    print("\nDX mean abs:")
    print("CasADi:", np.mean(np.abs(dx_ca), axis=0))
    print("Torch :", np.mean(np.abs(dx_th), axis=0))
    print("Diff  :", np.mean(np.abs(dx_diff), axis=0))

    print("\nDX max abs:")
    print("CasADi:", np.max(np.abs(dx_ca), axis=0))
    print("Torch :", np.max(np.abs(dx_th), axis=0))
    print("Diff  :", np.max(np.abs(dx_diff), axis=0))

    print("\nSmall z comparison:")
    for j, name in enumerate(Z_NAMES_SMALL):
        print(
            f"{name:12s} | "
            f"mean abs CasADi = {np.mean(np.abs(z_ca[:, j])):.6e} | "
            f"mean abs Torch = {np.mean(np.abs(z_th[:, j])):.6e} | "
            f"mean abs diff = {np.mean(np.abs(z_diff[:, j])):.6e} | "
            f"max abs diff = {np.max(np.abs(z_diff[:, j])):.6e}"
        )

    print("\nDetailed pointwise small z comparison:")
    for k in range(U_17.shape[0]):
        print(
            f"k={k:02d} u=({U_17[k,0]:.6f}, {U_17[k,1]:.6f}) | "
            f"CasADi z={z_ca[k]} | "
            f"Torch z={z_th[k]} | "
            f"diff={z_diff[k]}"
        )

    return {
        "U_17": U_17,
        "Y_17": Y_17,
        "dx_casadi": dx_ca,
        "dx_torch": dx_th,
        "dx_diff": dx_diff,
        "z_casadi": z_ca,
        "z_torch": z_th,
        "z_diff": z_diff,
    }


def compare_dx_on_17_points(well_name):
    """
    Load poly_dataset.pkl and compare CasADi dx vs Torch dx at the 17 selected points.
    """

    folder = get_well_folder(well_name)
    dataset_path = folder / "poly_dataset.pkl"

    poly_dataset = load_pickle(dataset_path)

    U_17, Y_17, state_names = get_17_state_data(poly_dataset)

    dx_ca, z_ca = evaluate_dx_casadi_17_points(
        well_name=well_name,
        U_17=U_17,
        Y_17=Y_17,
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr
    )

    dx_torch,z_torch = evaluate_dx_torch_17_points(
        well_name=well_name,
        U_17=U_17,
        Y_17=Y_17,
        BSW=BSW,
        GOR=GOR,
        PI=PI,
        K_gs=K_gs,
        K_inj=K_inj,
        K_pr=K_pr,
    )

    dx_diff = dx_torch - dx_ca

    rows = []

    for k in range(U_17.shape[0]):
        rows.append({
            "k": k,
            "u1": U_17[k, 0],
            "u2": U_17[k, 1],

            "dx1_casadi": dx_ca[k, 0],
            "dx2_casadi": dx_ca[k, 1],
            "dx3_casadi": dx_ca[k, 2],

            "dx1_torch": dx_torch[k, 0],
            "dx2_torch": dx_torch[k, 1],
            "dx3_torch": dx_torch[k, 2],

            "dx1_diff": dx_diff[k, 0],
            "dx2_diff": dx_diff[k, 1],
            "dx3_diff": dx_diff[k, 2],

            "norm_dx_casadi": np.linalg.norm(dx_ca[k, :]),
            "norm_dx_torch": np.linalg.norm(dx_torch[k, :]),
            "norm_diff": np.linalg.norm(dx_diff[k, :]),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"Residual comparison on 17 points for well {well_name}")
    print(f"State names: {state_names}")
    print("=" * 100)

    print("\nCasADi raw residual summary:")
    print("mean abs:", np.mean(np.abs(dx_ca), axis=0))
    print("max  abs:", np.max(np.abs(dx_ca), axis=0))

    print("\nTorch raw residual summary:")
    print("mean abs:", np.mean(np.abs(dx_torch), axis=0))
    print("max  abs:", np.max(np.abs(dx_torch), axis=0))

    print("\nTorch - CasADi residual difference summary:")
    print("mean abs:", np.mean(np.abs(dx_diff), axis=0))
    print("max  abs:", np.max(np.abs(dx_diff), axis=0))

    print("\nDetailed residual table:")
    print(df.to_string(index=False))

    return {
        "U_17": U_17,
        "Y_17": Y_17,
        "dx_casadi": dx_ca,
        "dx_torch": dx_torch,
        "dx_diff": dx_diff,
        "df": df,
    }



if __name__ == "__main__":

    WELL_NAME="P1"
    wells = get_wells()
    w = wells[WELL_NAME]
    BSW=w["BSW"]
    GOR=w["GOR"]
    PI=w["PI"]
    K_gs=w["K_gs"]
    K_inj=w["K_inj"]
    K_pr=w["K_pr"]
    result = compare_dx_on_17_points(WELL_NAME)

    result2=compare_small_z_on_17_points(WELL_NAME)
