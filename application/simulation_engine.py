# THIS SIMULATOR HAS ACCESS TO THE BLACK BOX MODEL
from simulators.black_box_simulator.black_box_model import make_glc_well_rigorous
# AND ALSO TO THE SURROGATE MODEL (in CasADi, simulation only)
from simulators.surrogate_simulator.surrogate_model_casadi import make_glc_well_surrogate
#FOR BOTH, Z_NAMES IS USED AS A STANDARD LIST OF VARIABLES
from simulators.Z_NAMES import Z_NAMES

# BUILDING THE MODEL IS ALSO A NECESSARY FOUNDATION BLOCK
from utilities.block_builders import build_steady_state_model

# AS WELL AS THE STEADY-STATE EQULIBRIUM SOLVER
from solvers.steady_state_solver import solve_equilibrium_ipopt

# NUMPY IS USED AS WELL
import numpy as np


################### MAKE MODEL ###################

def make_model(sim_kind:str,
               BSW,  #0.20
               GOR,  # 0.05
               PI, #3.0e-6
               K_gs,
               K_inj,
               K_pr):
    sim_kind = sim_kind.lower().strip()

    if sim_kind == "surrogate":
        well_func_sur = make_glc_well_surrogate(BSW=BSW,
                                                GOR=GOR,
                                                PI=PI,
                                                K_gs=K_gs,
                                                K_inj=K_inj,
                                                K_pr=K_pr)
        model=build_steady_state_model(
            f_func=well_func_sur,
            state_size=3,
            control_size=2,
            alg_size=None,
            name="glc_surrogate_ss",
            out_name=Z_NAMES
        )
        return model
    elif sim_kind == "rigorous":
        well_func_rig = make_glc_well_rigorous(BSW=BSW,
                                               GOR=GOR,
                                               PI=PI,
                                               K_gs=K_gs,
                                               K_inj=K_inj,
                                               K_pr=K_pr)
        model=build_steady_state_model(
            f_func=well_func_rig,
            state_size=7,
            control_size=2,
            alg_size=4,
            name="glc_rigorous_ss",
            out_name=Z_NAMES
        )
        return model
    else:
        raise ValueError("sim_kind must be 'surrogate' or 'rigorous'")

################### RUN SWEEP ###################

def run_sweep(model,
              U1_MIN,
              U2_MIN,
              U_SIM_SIZE,
              y_guess_init,
              z_guess_init=None,
              RES_TOL_DX=1e-6,
              RES_TOL_G=1e-6,
              TOL_EIG=1e-8):

    u1_grid = np.linspace(U1_MIN, 1.00001, U_SIM_SIZE)
    u2_grid = np.linspace(U2_MIN, 1.00001, U_SIM_SIZE)

    U1,U2=np.meshgrid(u1_grid,u2_grid,indexing='ij')

    nx=model["nx"]
    nu=model["nu"]
    is_dae=bool(model["is_dae"])

    Z_NAMES=model["Z_NAMES"]
    n_out=len(Z_NAMES)

    Nu1=len(u1_grid)
    Nu2=len(u2_grid)

    OUT = {name: np.full((Nu1, Nu2), np.nan, dtype=float) for name in Z_NAMES}

    RES_DX= np.full((Nu1, Nu2), np.nan, dtype=float)
    RES_G=np.full((Nu1, Nu2), np.nan, dtype=float) if is_dae else None

    STABLE = np.full((Nu1, Nu2), np.nan, dtype=float)  # 1 stable, 0 unstable, NaN unknown/fail
    SUCCESS = np.zeros((Nu1, Nu2), dtype=bool)
    P_max=np.zeros((Nu1,Nu2),dtype=float)

    i_iter = range(Nu1 - 1, -1, -1)
    j_iter = range(Nu2 - 1, -1, -1)
    j_start=Nu2-1

    prev_row_rightmost_y = None
    prev_row_rightmost_z = None  # only used if is_dae

    for i in i_iter:
        u1 = u1_grid[i]
        if prev_row_rightmost_y is not None:
            y_guess = prev_row_rightmost_y
            if is_dae:
                z_guess = prev_row_rightmost_z
        else:
            y_guess=np.array(y_guess_init,dtype=float).reshape(-1)
            z_guess = None if (not is_dae) else np.array(z_guess_init, dtype=float).reshape(-1)

        for j in j_iter:
            u2 = u2_grid[j]
            print("\n----------------------------------")
            print(f"u1={u1} u2={u2}")
            try:
                if is_dae:
                    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                        model=model,
                        u_val=[u1, u2],
                        y_guess=y_guess,
                        z_guess=z_guess,
                        tol_eig=TOL_EIG
                    )
                else:
                    y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                    model=model,
                    u_val=[u1, u2],
                    y_guess=y_guess,
                    tol_eig=TOL_EIG
                    )

                print("IPOPT success:", bool(stats.get("success", False)))
                print("IPOPT status :", stats.get("return_status", ""))

                # if not bool(stats.get("success", False)):
                #     continue

                dx_np = np.array(dx_star, dtype=float).reshape(-1)
                res_dx=float(np.linalg.norm(dx_np))
                RES_DX[i,j]=res_dx


                if is_dae:
                    g_np=np.array(g_star, dtype=float).reshape(-1)
                    res_g=float(np.linalg.norm(g_np))
                    RES_G[i,j]=res_g

                    if (res_dx > RES_TOL_DX) or (res_g > RES_TOL_G):

                        print(f"High residuals: ||dx||={res_dx:.3e}, ||g||={res_g:.3e}")
                        continue
                else:
                    if res_dx > RES_TOL_DX:
                        print(f"High residual: ||dx||={res_dx:.3e}")
                        continue

                out_np = np.array(out_star, dtype=float).reshape(-1)
                if out_np.size != n_out:
                    print(f"WARNING: out size {out_np.size} != Z_NAMES {n_out}. Not storing OUT for this point.")
                else:
                    for k, name in enumerate(Z_NAMES):
                        OUT[name][i, j] = out_np[k]

                STABLE[i, j] = 1.0 if stable else 0.0
                SUCCESS[i, j] = True
                P_max[i,j]=90

                if stable:
                    y_guess=np.array(y_star,dtype=float).reshape(-1)
                    if is_dae:
                        z_guess=np.array(z_star,dtype=float).reshape(-1)
                        print("Accepted z*:", z_star,"||g||",res_g)
                if j==j_start and stable:
                    prev_row_rightmost_y = np.array(y_star, dtype=float).reshape(-1)
                    if is_dae:
                        prev_row_rightmost_z = np.array(z_star, dtype=float).reshape(-1)
                print("Accepted. y*:",y_star,"||dx||:", res_dx, "stable:", stable)
            except Exception as e:
                print("Exception:", repr(e))
                continue
    return {
        "OUT": OUT,
        "RES_DX": RES_DX,
        "RES_G":RES_G,
        "STABLE": STABLE,
        "SUCCESS": SUCCESS,
        "u1_grid": u1_grid,
        "u2_grid": u2_grid,
        "U1": U1,
        "U2": U2,
        "Z_NAMES": Z_NAMES,
        "is_dae": is_dae,
        "P_max": P_max
    }

################### EXTRACT STABILITY BOUNDARY FROM GRID ###################

def extract_stability_boundary_from_grid(
        u1_grid,
        u2_grid,
        STABLE
):
    u1_grid=np.asarray(u1_grid,dtype=float).reshape(-1)
    u2_grid=np.asarray(u2_grid,dtype=float).reshape(-1)
    STABLE=np.asarray(STABLE,dtype=float)

    Nu1=len(u1_grid)
    Nu2=len(u2_grid)

    if STABLE.shape != (Nu1,Nu2):
        raise ValueError(f"STABLE shape {STABLE.shape} does not match (Nu1,Nu2)=({Nu1,Nu2})")

    boundary_u1=[]
    boundary_u2=[]

    for i,u1v in enumerate(u1_grid):
        stable_js=np.where(STABLE[i,:]==1.0)[0]
        if stable_js.size > 0:
            jmin=stable_js.min()
            boundary_u1.append(u1v)
            boundary_u2.append(u2_grid[jmin])

    return np.asarray(boundary_u1),np.asarray(boundary_u2)

################### FIT POLYNOMIAL BOUNDARY ###################

def fit_boundary_polynomial(boundary_u1,
                            boundary_u2,
                            deg=2):
    boundary_u1=np.asarray(boundary_u1,dtype=float).reshape(-1)
    boundary_u2=np.asarray(boundary_u2,dtype=float).reshape(-1)

    if boundary_u1.size < deg+1:
        raise RuntimeError(f"Need at least {deg+1} boundary points, got {boundary_u1.size}")
    idx=np.argsort(boundary_u1)
    boundary_u1=boundary_u1[idx]
    boundary_u2=boundary_u2[idx]

    coef=np.polyfit(boundary_u1,boundary_u2,deg=deg)
    b_hat=np.poly1d(coef)
    return b_hat

import torch



@torch.no_grad()
def run_sweep_PINN(
    model,
    U1_MIN,
    U2_MIN,
    U1_MAX=1.0,
    U2_MAX=1.0,
    U_SIM_SIZE=20):
    """
    Returns:
      U1, U2 : meshgrids shape (n_u2, n_u1)
      Y      : pinn outputs shape (n_u2, n_u1, 3)
      Z      : algnn outputs shape (n_u2, n_u1, 3) -> [m_o_out, p_bh, p_tb_b]
    """
    u1 = np.linspace(U1_MIN, U1_MAX, U_SIM_SIZE, dtype=np.float32)
    u2 = np.linspace(U2_MIN, U2_MAX, U_SIM_SIZE, dtype=np.float32)
    U1, U2 = np.meshgrid(u1, u2, indexing="ij")

    # Evaluate one point to get output size
    z0 = np.array(model(u=np.array([u1[0], u2[0]]))["z"]).reshape(-1)
    n_z = len(z0)

    Z = np.zeros((U_SIM_SIZE, U_SIM_SIZE, n_z))

    for i in range(U_SIM_SIZE):
        for j in range(U_SIM_SIZE):
            uk = np.array([U1[i, j], U2[i, j]])
            zk = np.array(model(u=uk)["z"]).reshape(-1)
            Z[i, j, :] = zk

    return U1, U2, Z


import numpy as np

def extract_threshold_boundary_from_grid(
    u1_grid,
    u2_grid,
    Z,
    threshold,
    mode=">=",
    side="first_true",
):
    """
    Extract a threshold boundary from a 2D grid Z(u1,u2).

    Parameters
    ----------
    u1_grid : array-like, shape (Nu1,)
    u2_grid : array-like, shape (Nu2,)
    Z : array-like, shape (Nu1, Nu2)
    threshold : float
    mode : str
        One of [">=", "<=", ">", "<"].
    side : str
        "first_true" -> first u2 where condition is satisfied
        "last_true"  -> last  u2 where condition is satisfied

    Returns
    -------
    boundary_u1 : ndarray
    boundary_u2 : ndarray
    """
    u1_grid = np.asarray(u1_grid, dtype=float).reshape(-1)
    u2_grid = np.asarray(u2_grid, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)

    Nu1 = len(u1_grid)
    Nu2 = len(u2_grid)

    if Z.shape != (Nu1, Nu2):
        raise ValueError(f"Z shape {Z.shape} does not match (Nu1,Nu2)=({Nu1},{Nu2})")

    if mode == ">=":
        cond = Z >= threshold
    elif mode == "<=":
        cond = Z <= threshold
    elif mode == ">":
        cond = Z > threshold
    elif mode == "<":
        cond = Z < threshold
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    boundary_u1 = []
    boundary_u2 = []

    for i, u1v in enumerate(u1_grid):
        js = np.where(cond[i, :])[0]
        if js.size == 0:
            continue

        if side == "first_true":
            j_sel = js.min()
        elif side == "last_true":
            j_sel = js.max()
        else:
            raise ValueError(f"Unsupported side: {side}")

        boundary_u1.append(u1v)
        boundary_u2.append(u2_grid[j_sel])

    return np.asarray(boundary_u1), np.asarray(boundary_u2)



