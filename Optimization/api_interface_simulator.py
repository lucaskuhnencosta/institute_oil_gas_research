# api_interface_simulator.py
import numpy as np
import casadi as ca

from Solvers.solve_glc_ode_equilibrium import solve_equilibrium_ipopt
from Application.model_analysis_application import make_model

# Indices you gave (0-based indexing assumed!)
IDX_P_BH = 15
IDX_P_TB_B = 17
IDX_W_O = 38


def _as_dm(x):
    return x if isinstance(x, (ca.DM, ca.MX, ca.SX)) else ca.DM(x)


def plant_zeroth_and_first_order(
    model,
    u_k,
    y_guess,
    z_guess,
    *,
    out_indices=(IDX_W_O, IDX_P_BH, IDX_P_TB_B),
):
    """
    Computes:
      z0 = plant outputs at u_k  (0th order)
      J  = dz/du at u_k          (1st order)

    For DAE steady-state:
      F(x,u)=0 where x=[y;z]
      z = h(x,u) = selected entries from out(y,z,u)

    Requires model["F_all"](y,z,u) -> dx,g,out  (DAE case).

    Returns dict with:
      y_star, z_star, z0, J, stats, stable, eig
    """
    u_k = np.array(u_k, dtype=float).reshape((-1,))
    nu = int(model["nu"])
    assert u_k.size == nu, f"u_k has size {u_k.size} but model nu={nu}"

    is_dae = bool(model["is_dae"])
    if not is_dae:
        raise ValueError("This API is for the rigorous DAE plant. Your model says is_dae=False.")

    # ---------- 1) Solve equilibrium at u_k ----------
    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
        model=model,
        u_val=u_k,
        y_guess=y_guess,
        z_guess=z_guess,
    )
    dx_np = np.array(dx_star, dtype=float).reshape(-1)
    res_dx = float(np.linalg.norm(dx_np))
    if is_dae:
        g_np = np.array(g_star, dtype=float).reshape(-1)
    res_g = float(np.linalg.norm(g_np))

    RES_TOL_DX=1e-6
    RES_TOL_G=1e-6
    if (res_dx > RES_TOL_DX) or (res_g > RES_TOL_G):
        raise RuntimeError(f"Equilibrium solve failed: {stats.get('return_status','')}")



    # out_star may be DM or numpy; convert to flat numpy
    out_np = np.array(out_star, dtype=float).reshape((-1,))

    # ---------- 2) 0th order term: z0 = [w_o_out, P_bh_bar, P_tb_b_bar] ----------
    z0 = np.array([out_np[i] for i in out_indices], dtype=float)

    # ---------- 3) Build implicit sensitivity dz/du ----------
    # Need symbolic residuals and outputs from the model
    F_all = model["F_all"]

    nx = int(model["nx"])
    nz = int(model["nz"])

    y_sym = ca.MX.sym("y", nx)
    z_sym = ca.MX.sym("z", nz)
    u_sym = ca.MX.sym("u", nu)

    dx_sym, g_sym, out_sym = F_all(y_sym, z_sym, u_sym)

    # F(x,u)=0
    F_sym = ca.vertcat(dx_sym, g_sym)         # (nx+nz,)

    # h(x,u) = selected outputs
    h_sym = ca.vertcat(*[out_sym[i] for i in out_indices])  # (3,)

    # Jacobians
    x_sym = ca.vertcat(y_sym, z_sym)
    Fx = ca.jacobian(F_sym, x_sym)    # (nx+nz) x (nx+nz)
    Fu = ca.jacobian(F_sym, u_sym)    # (nx+nz) x nu
    hx = ca.jacobian(h_sym, x_sym)    # 3 x (nx+nz)
    hu = ca.jacobian(h_sym, u_sym)    # 3 x nu

    # Evaluate at solution
    y_dm = _as_dm(y_star)
    z_dm = _as_dm(z_star)
    u_dm = _as_dm(u_k)

    Fx_num = ca.DM(ca.Function("Fx_f", [y_sym, z_sym, u_sym], [Fx])(y_dm, z_dm, u_dm))
    Fu_num = ca.DM(ca.Function("Fu_f", [y_sym, z_sym, u_sym], [Fu])(y_dm, z_dm, u_dm))
    hx_num = ca.DM(ca.Function("hx_f", [y_sym, z_sym, u_sym], [hx])(y_dm, z_dm, u_dm))
    hu_num = ca.DM(ca.Function("hu_f", [y_sym, z_sym, u_sym], [hu])(y_dm, z_dm, u_dm))

    # Solve for dx*/du:  dxdu = -Fx^{-1} Fu
    # (use solve instead of explicit inverse)
    dxdu = -ca.solve(Fx_num, Fu_num)           # (nx+nz) x nu

    # dz/du = hx*dxdu + hu
    J = hx_num @ dxdu + hu_num                 # 3 x nu

    return {
        "u_k": u_k,
        "y_star": np.array(y_star, dtype=float).reshape((-1,)),
        "z_star": np.array(z_star, dtype=float).reshape((-1,)),
        "z0": np.array(z0, dtype=float),               # [w_o_out, P_bh, P_tb_b]
        "J": np.array(J, dtype=float),                 # 3x2
        "stable": stable,
        "eig": eig,
        "stats": stats,
    }


if __name__ == "__main__":
    from Rigorous_DAE_model.glc_truth_casadi import make_glc_well_rigorous, Z_NAMES_RIG
    from Utilities.block_builders import *

    Z_NAMES=Z_NAMES_RIG


    u_k = [0.71003869, 0.64829166]
    well_func = make_glc_well_rigorous(BSW=0.20, GOR=0.05, PI=3.0e-6)
    model_rigorous = build_steady_state_model(
        f_func=well_func,
        state_size=7,
        control_size=2,
        alg_size=4,
        name="glc_ss_rigorous",
        out_name=Z_NAMES
    )

    y_guess_rig = [3679.08033973,
               289.73390193,
               3167.56224658,
               1041.96126532,
               50.46858403,
               759.52720527,
               249.84447542]

    z_guess_rig = [8.75897957e+06,
               8.42155186e+06,
               2.17230613e+01,
               2.17230613e+01]
    # provide good guesses (e.g. last plant solution)
    y_guess = y_guess_rig
    z_guess = z_guess_rig

    res = plant_zeroth_and_first_order(model_rigorous, u_k, y_guess, z_guess)

    np.set_printoptions(suppress=True, precision=6)

    print("Plant z0(u_k) = [w_o_out, P_bh_bar, P_tb_b_bar]:")
    print(res["z0"])

    print("\nPlant J(u_k) = dz/du:")
    print(res["J"])