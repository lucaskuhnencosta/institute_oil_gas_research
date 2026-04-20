# api_interface_simulator.py

import numpy as np
import casadi as ca
from sympy.polys.numberfields import utilities

from solvers.steady_state_solver import solve_equilibrium_ipopt
from simulators.Z_NAMES import Z_NAMES


def _as_dm(x):
    return x if isinstance(x, (ca.DM, ca.MX, ca.SX)) else ca.DM(x)


def plant_zeroth_and_first_order(
    model,
    u_k,
    y_guess,
    z_guess,
    out_names,
):
    """
    Single-well rigorous plant API.

    Parameters
    ----------
    model : dict
        Well DAE model dictionary.
    u_k : array-like, shape (nu,)
        Current control input for this well.
    y_guess : array-like
        Initial guess for differential states.
    z_guess : array-like
        Initial guess for algebraic states.
    out_indices : list[int]
        Indices of the plant outputs to keep in z0.

    Returns
    -------
    dict with keys:
        y_star : ndarray
        z_star : ndarray
        z0     : ndarray   selected plant outputs
        J      : ndarray   dz/du for selected outputs
    """

    out_indices=[Z_NAMES.index(name) for name in out_names]

    # Convert the control to a flat Numpy vector
    u_k = np.array(u_k, dtype=float).reshape((-1,))

    # ==========================================================
    # 1) Solve the steady-state DAE equilibrium at u_k
    # ==========================================================
    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
        model=model,
        u_val=u_k,
        y_guess=y_guess,
        z_guess=z_guess,
    )
    # Residuals of the solved equilibrium.
    # dx_star corresponds to differential equations residuals.
    # g_star corresponds to algebraic equations residuals.
    dx_np = np.array(dx_star, dtype=float).reshape(-1)
    g_np = np.array(g_star, dtype=float).reshape(-1)


    res_dx = float(np.linalg.norm(dx_np))
    res_g = float(np.linalg.norm(g_np))

    # Tolerances for declaring the equilibrium properly solved.
    RES_TOL_DX = 1e-6
    RES_TOL_G = 1e-6
    if (res_dx > RES_TOL_DX) or (res_g > RES_TOL_G):
        raise RuntimeError(f"Equilibrium solve failed: {stats.get('return_status','')}")

    # Convert the full plant output vector to flat NumPy.
    out_np = np.array(out_star, dtype=float).reshape((-1,))

    # print("out_np: ", out_np)

    # ==========================================================
    # 2) Zeroth-order term z0 = selected plant outputs
    # ==========================================================
    # Example:
    # z0 = [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]
    z0 = np.array([out_np[i] for i in out_indices], dtype=float)


    # ==========================================================
    # 3) Implicit sensitivity: J = dz/du
    # ==========================================================
    # We use the DAE steady-state equations:
    #
    #     F(x,u) = 0
    #
    # where x = [y; z]
    #
    # and the selected outputs:
    #
    #     h(x,u) = selected outputs
    #
    # Then:
    #
    #     dx*/du = - Fx^{-1} Fu
    #     dz/du  = hx (dx*/du) + hu
    #
    F_all = model["F_all"]

    nu = int(model["nu"])
    nx = int(model["nx"])
    nz = int(model["nz"])

    y_sym = ca.MX.sym("y", nx)
    z_sym = ca.MX.sym("z", nz)
    u_sym = ca.MX.sym("u", nu)

    dx_sym, g_sym, out_sym = F_all(y_sym, z_sym, u_sym)

    # F(x,u)=0
    F_sym = ca.vertcat(dx_sym, g_sym)         # (nx+nz,)

    # Select only the outputs of interest.
    h_sym = ca.vertcat(*[out_sym[i] for i in out_indices])

    # Build symbolic stacked state vector x = [y; z].
    x_sym = ca.vertcat(y_sym, z_sym)

    # Jacobians needed by implicit differentiation
    Fx = ca.jacobian(F_sym, x_sym)    # (nx+nz) x (nx+nz)
    Fu = ca.jacobian(F_sym, u_sym)    # (nx+nz) x nu
    hx = ca.jacobian(h_sym, x_sym)    # 3 x (nx+nz)
    hu = ca.jacobian(h_sym, u_sym)    # 3 x nu

    # Evaluate all Jacobians at the solved equilibrium.
    y_dm = _as_dm(y_star)
    z_dm = _as_dm(z_star)
    u_dm = _as_dm(u_k)

    Fx_num = ca.DM(ca.Function("Fx_f", [y_sym, z_sym, u_sym], [Fx])(y_dm, z_dm, u_dm))
    Fu_num = ca.DM(ca.Function("Fu_f", [y_sym, z_sym, u_sym], [Fu])(y_dm, z_dm, u_dm))
    hx_num = ca.DM(ca.Function("hx_f", [y_sym, z_sym, u_sym], [hx])(y_dm, z_dm, u_dm))
    hu_num = ca.DM(ca.Function("hu_f", [y_sym, z_sym, u_sym], [hu])(y_dm, z_dm, u_dm))

    # Solve linear system for dx*/du.
    # This is better than explicitly inverting Fx.
    dxdu = -ca.solve(Fx_num, Fu_num)           # (nx+nz) x nu

    # dz/du = hx*dxdu + hu
    J = hx_num @ dxdu + hu_num                 # 3 x nu

    return {
        "y_star": np.array(y_star, dtype=float).reshape((-1,)),
        "z_star": np.array(z_star, dtype=float).reshape((-1,)),
        "z0": np.array(z0, dtype=float),               # [w_o_out, P_bh, P_tb_b]
        "J": np.array(J, dtype=float),                 # 3x2
    }


if __name__ == "__main__":
    from utilities.block_builders import *
    from configuration.wells import get_wells
    from application.simulation_engine import make_model
    from configuration.surrogate_based_optimizer_configs import get_solver_configs

    wells = get_wells()
    config=get_solver_configs()
    model_type = "rigorous"

    well_names = list(wells.keys())
    N = len(well_names)


    u_k = np.array(config["u_guess_list"].reshape(-1))
    print(f"at this point, uk is {u_k} and its shape is {u_k.shape}")

    for j, well_name in enumerate(well_names):
        well_data = wells[well_name]
        model = make_model(
                model_type,
                BSW=well_data["BSW"],
                GOR=well_data["GOR"],
                PI=well_data["PI"],
                K_gs=well_data["K_gs_sur"],
                K_inj=well_data["K_inj_sur"],
                K_pr=well_data["K_pr_sur"]
        )

        # Slice the stacked field control to obtain the local well control
        start = j * 2
        end = (j + 1) *2
        u_j = u_k[start:end]
        print(f"at this point, uj is {u_j} and its shape is {u_j.shape}")

        y_guess = well_data["y_guess_rig"]
        z_guess = well_data["z_guess_rig"]

        res_j = plant_zeroth_and_first_order(
            model=model,
            u_k=u_j,
            y_guess=y_guess,
            z_guess=z_guess,
            out_names=config["out_names"]
        )

        print(res_j)
