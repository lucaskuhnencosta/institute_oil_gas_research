from Surrogate_ODE_Model.glc_coarse_casadi import glc_casadi
from Surrogate_ODE_Model.glc_surrogate_casadi import glc_well_01_casadi
from Rigorous_DAE_model.glc_rigorous_casadi import glc_well_01_rigorous_casadi
from Utilities.block_builders import build_steady_state_model
import casadi as ca
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

def solve_equilibrium_ipopt(
        model, # Ourput of a CasADi model (steady-state) that has been assembled
        u_val, #list/array shape (nu,)
        y_guess, #list/array shape (nx,)
        z_guess=None
        ):
    """
    Works for both:
      - ODE surrogate model dict from build_steady_state_model_unified(..., alg_size=None)
      - DAE rigorous model dict from build_steady_state_model_unified(..., alg_size=3)

    Decision variables:
      ODE: x = y
      DAE: x = [y; z_alg]

    Constraints:
      ODE: dx(y,u)=0
      DAE: dx(y,z,u)=0 and g(y,z,u)=0

    Stability:
      ODE: eig(A) where A = d(dx)/d(y)
      DAE: eig(Ared) where Ared = fx - fz * (gz \\ gx)  (index-1 reduction)
    """

    # ---------------------
    # 1) Unpack model pieces
    # ---------------------
    nx=model["nx"]
    nu=model["nu"]
    is_dae=bool(model["is_dae"])

    y_sym=model["y"]
    u_sym = model["u"]
    Z_NAMES = model["Z_names"]

    # ---------------------
    # 2) Decision variables
    # ---------------------

    y_var = ca.MX.sym("y_var", nx)

    if is_dae:
        nz = model["nz"]
        z_sym = model["z_alg"]
        z_var = ca.MX.sym("z_var", nz)
        x_var = ca.vertcat(y_var, z_var)
    else:
        x_var=y_var

    # ---------------------
    # 3) Parameter (fixed control)
    # ---------------------
    u_par = ca.MX.sym("u_par", nu)

    # ---------------------
    # 4) Model call (use model["F_all"] directly)
    # ---------------------
    if is_dae:
        dx_var, g_var, out_var = model["F_all"](y_var, z_var, u_par)
    else:
        dx_var, out_var = model["F_all"](y_var, u_par)

    Zsym = {name: out_var[i] for i, name in enumerate(Z_NAMES)}

    # ---------------------
    # 5) Unpack variables
    # ---------------------

    P_an_t_bar = Zsym["P_an_t_bar"]
    P_an_b_bar = Zsym["P_an_b_bar"]
    rho_G_an_b = Zsym["rho_G_an_b"]
    rho_G_in = Zsym["rho_G_in"]

    dP_gs_an_bar = Zsym["dP_gs_an_bar"]
    w_G_in_original = Zsym["w_G_in_original"]
    w_G_in = Zsym["w_G_in"]

    V_gas_tb_t = Zsym["V_gas_tb_t"]
    V_gas_tb_t_safe = Zsym["V_gas_tb_t_safe"]
    rho_G_tb_t = Zsym["rho_G_tb_t"]
    P_tb_t_bar = Zsym["P_tb_t_bar"]

    rho_avg_mix_tb = Zsym["rho_avg_mix_tb"]
    alpha_avg_L_tb = Zsym["alpha_avg_L_tb"]
    alpha_G_tb_b = Zsym["alpha_G_m_tb_b"]
    U_avg_L_tb = Zsym["U_avg_L_tb"]
    denom_G = Zsym["denom_G"]
    denom_G_safe = Zsym["denom_G_safe"]
    U_avg_G_tb = Zsym["U_avg_G_tb"]
    U_avg_mix_tb = Zsym["U_avg_mix_tb"]

    Re_tb = Zsym["Re_tb"]
    Re_tb_safe = Zsym["Re_tb_safe"]
    log_arg_tb = Zsym["log_arg_tb"]
    log_arg_tb_safe = Zsym["log_arg_tb_safe"]
    lambda_tb = Zsym["lambda_tb"]
    F_t_bar = Zsym["F_t_bar"]

    P_tb_b_bar = Zsym["P_tb_b_bar"]
    dP_an_tb_bar = Zsym["dP_an_tb_bar"]
    w_G_inj = Zsym["w_G_inj"]

    U_avg_L_bh = Zsym["U_avg_L_bh"]
    Re_bh = Zsym["Re_bh"]
    log_arg_bh = Zsym["log_arg_bh"]
    lambda_bh = Zsym["lambda_bh"]
    F_bh_bar = Zsym["F_bh_bar"]
    P_bh_bar = Zsym["P_bh_bar"]

    dP_res_bh_bar = Zsym["dP_res_bh_bar"]
    w_res = Zsym["w_res"]
    w_L_res = Zsym["w_L_res"]
    w_G_res = Zsym["w_G_res"]
    rho_G_tb_b = Zsym["rho_G_tb_b"]

    denom_alpha_b = Zsym["denom_alpha_b"]
    denom_alpha_b_safe = Zsym["denom_alpha_b_safe"]
    alpha_L_tb_b = Zsym["alpha_L_tb_b"]
    alpha_L_tb_t = Zsym["alpha_L_tb_t"]
    rho_mix_tb_t = Zsym["rho_mix_tb_t"]
    rho_mix_tb_t_safe = Zsym["rho_mix_tb_t_safe"]

    dP_tb_choke_bar = Zsym["dP_tb_choke_bar"]
    w_out = Zsym["w_out"]
    Q_out = Zsym["Q_out"]
    denom_alpha_t = Zsym["denom_alpha_t"]
    denom_alpha_t_safe = Zsym["denom_alpha_t_safe"]
    alpha_G_tb_t = Zsym["alpha_G_tb_t"]
    w_G_out = Zsym["w_G_out"]
    w_L_out = Zsym["w_L_out"]

    # ---------------------
    # 6) Objective: minimize ||dx||^2 (scaled)
    # ---------------------
    obj=ca.dot(dx_var,dx_var)
    p=u_par

    # ---------------------
    # 7) Constraints
    # ---------------------
    g_list=[]
    lbg=[]
    ubg=[]

    # Equality constraints: dx = 0 (nx) and g = 0 (nz)
    g_list.append(dx_var)
    lbg.extend([0] * nx)
    ubg.extend([0] * nx)

    if is_dae:
        g_list.append(g_var)
        lbg.extend([0] * nz)
        ubg.extend([0] * nz)

    g_list.append(log_arg_tb_safe)
    lbg.append(1e-12)
    ubg.append(1e20)

    g_list.append(log_arg_bh)
    lbg.append(1e-12)
    ubg.append(1e20)

    for a in [alpha_L_tb_b, alpha_L_tb_t, alpha_G_tb_t, alpha_avg_L_tb]:
        g_list.append(a)
        lbg.append(0.0)
        ubg.append(1.0)

    for pbar in [P_an_t_bar, P_an_b_bar, P_tb_t_bar, P_tb_b_bar, P_bh_bar]:
        g_list.append(pbar)
        lbg.append(1e-6)  # >0 bar
        ubg.append(1e20)

    # production choke forward
    g_list.append(dP_tb_choke_bar)
    lbg.append(0)
    ubg.append(1e20)

    # gas source -> annulus
    g_list.append(dP_gs_an_bar)
    lbg.append(0)
    ubg.append(1e20)

    # reservoir -> well
    g_list.append(dP_res_bh_bar)
    lbg.append(0)
    ubg.append(1e20)

    # instead of dP_an_tb_bar >= 0, enforce nonnegative injection (already true)
    g_list.append(w_G_inj)
    lbg.append(0)
    ubg.append(1e20)

    g_list.append(w_out)
    lbg.append(0)
    ubg.append(1e20)

    g_list.append(w_L_out)
    lbg.append(0)
    ubg.append(1e20)

    g_list.append(w_G_out)
    lbg.append(0)
    ubg.append(1e20)

    g_nlp=ca.vertcat(*g_list)

    # ---------------------
    # 8) Bounds on y
    # ---------------------
    y_lb=[0.0,0.0,0.0]
    y_ub=[1e20,1e20,1e20]

    # ---------------------
    # 9) Bounds on z
    # ---------------------
    if is_dae:
        z_lb=[1e3,1e3,0.0]
        z_ub=[1e9,1e9,1e6]
        lbx=ca.DM(y_lb+z_lb).reshape((nx+nz,1))
        ubx=ca.DM(y_ub+z_ub).reshape((nx+nz,1))
    else:
        lbx=ca.DM(y_lb).reshape((nx,1))
        ubx=ca.DM(y_ub).reshape((nx,1))

    # ---------------------
    # 10) NLP definition
    # ---------------------
    nlp = {"x": x_var, "p": p, "f": obj, "g": g_nlp}

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 8000,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.tol": 1e-12,
        "ipopt.constr_viol_tol": 1e-10,
        "ipopt.dual_inf_tol": 1e-10,
        "ipopt.compl_inf_tol": 1e-10,
        "ipopt.acceptable_tol": 1e-20,
        "ipopt.acceptable_constr_viol_tol": 1e-20,
        "ipopt.acceptable_dual_inf_tol": 1e-20,
        "ipopt.acceptable_compl_inf_tol": 1e-20,
        "ipopt.acceptable_iter": 0,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    }

    solver = ca.nlpsol("eq_solver", "ipopt", nlp, opts)

    # -------------------------
    # 11) Pack numeric inputs
    # -------------------------
    u_val_dm = ca.DM(u_val).reshape((nu, 1))
    y0 = ca.DM(y_guess).reshape((nx, 1))

    if is_dae:
        if z_guess is None:
            raise ValueError("DAE model requires z_guess.(initial guess for algebraic variables).")
        z0=ca.DM(z_guess).reshape((nz, 1))
        x0=ca.vertcat(y0,z0)
    else:
        x0=y0

    # -------------------------
    # 12) Solve
    # -------------------------

    sol = solver(
        x0=x0,
        lbx=lbx,
        ubx=ubx,
        lbg=ca.DM(lbg) if len(lbg) else ca.DM([]),
        ubg=ca.DM(ubg) if len(ubg) else ca.DM([]),
        p=u_val_dm,
    )

    x_star = sol["x"]
    stats=solver.stats()

    if not bool(stats.get("success", False)):
        print("\n---- IPOPT FAILURE DIAGNOSTICS ----")
        print("Status:", stats.get("return_status", ""))

        x_last = sol["x"]

        if is_dae:
            y_last = x_last[0:nx]
            z_last = x_last[nx:nx + nz]
            dx_last, g_last, out_last = model["F_all"](y_last, z_last, u_val_dm)

            dx_np = np.array(dx_last, dtype=float).reshape(-1)
            g_np = np.array(g_last, dtype=float).reshape(-1)

            print("||dx|| =", np.linalg.norm(dx_np))
            print("||g||  =", np.linalg.norm(g_np))
            print("dx =", dx_np)
            print("g  =", g_np)

        else:
            y_last = x_last
            dx_last, out_last = model["F_all"](y_last, u_val_dm)

            dx_np = np.array(dx_last, dtype=float).reshape(-1)

            print("||dx|| =", np.linalg.norm(dx_np))
            print("dx =", dx_np)

        print("------------------------------------\n")

    # -------------------------
    # 13) Post-eval
    # -------------------------
    tol=1e-8

    if is_dae:
        y_star=x_star[0:nx]
        z_star=x_star[nx:nx+nz]
        dx_star,g_star,out_star=model["F_all"](y_star,z_star,u_val_dm)

        # Reduced Jacobian stability
        A_star = model["F_A"](y_star, z_star, u_val_dm)  # Ared
        A_num = np.array(A_star, dtype=float)
        eig = np.linalg.eigvals(A_num)
        stable = bool(np.all(np.real(eig) < -tol))

        return y_star, z_star, dx_star, g_star, out_star, eig, stable, stats

    else:
        y_star=x_star
        dx_star, out_star=model["F_all"](y_star,u_val_dm)

        # Reduced Jacobian stability
        A_star = model["F_A"](x_star, u_val_dm)
        A_num = np.array(A_star, dtype=float)
        eig = np.linalg.eigvals(A_num)
        stable = bool(np.all(np.real(eig) < -tol))

    return y_star, dx_star, out_star, eig, stable, stats


# model_surrogate = build_steady_state_model(glc_well_01_casadi,
#                                      state_size=3,
#                                      control_size=2)

model_rigorous=build_steady_state_model(glc_well_01_rigorous_casadi,
                                        state_size=3,
                                        alg_size=3,
                                        control_size=2)

u = [0.90, 0.90]
y_guess = [3309.01253915,252.30410395,7905.99065368]
z_guess = [120e5, 140e5, 10.0]  # [P_tb, P_bh, w_res] initial guesses (example)

# y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
#     model=model_surrogate,
#     u_val=u,
#     y_guess=y_guess,
# )

y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
    model=model_rigorous,
    u_val=u,
    y_guess=y_guess,
    z_guess=z_guess
)

print("status:", stats["return_status"], "success:", stats["success"])
print("y*:", np.array(y_star).squeeze())
print("dx*:", np.array(dx_star).squeeze())
print("||dx||:", np.linalg.norm(np.array(dx_star).squeeze()))
# print("z*:", np.array(z_star).squeeze())
print("\n--- z* (named) ---")
Z_NAMES=model_rigorous["Z_names"]
print_z_grouped(out_star, Z_NAMES)  # set ncols=1 if you prefer

