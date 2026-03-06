from Surrogate_ODE_Model.glc_surrogate_casadi import make_glc_well_surrogate
from Rigorous_DAE_model.glc_truth_casadi import make_glc_well_rigorous, Z_NAMES_RIG
from Utilities.block_builders import build_steady_state_model
import casadi as ca
import numpy as np
from collections import defaultdict

Z_NAMES=Z_NAMES_RIG

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
    # ---------------------
    # 1) Unpack model pieces
    # ---------------------
    nx=model["nx"]
    nu=model["nu"]
    is_dae=bool(model["is_dae"])

    y_sym=model["y"]
    u_sym = model["u"]
    Z_NAMES = model["Z_NAMES"]

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

    dP_tb_choke_bar = Zsym["dP_tb_choke_bar"]
    dP_gs_an_bar = Zsym["dP_gs_an_bar"]
    dP_res_bh_bar = Zsym["dP_res_bh_bar"]
    dP_int=Zsym["dP_int_bar"]

    w_G_inj=Zsym["w_G_inj"]

    m_o_t = Zsym["m_o_t"]
    m_w_t = Zsym["m_w_t"]
    m_o_b = Zsym["m_o_b"]
    m_w_b = Zsym["m_w_b"]
    w_up=Zsym["w_up"]

    w_out=Zsym["w_out"]
    w_res=Zsym["w_res"]

    alpha_L_tb=Zsym["alpha_L_tb"]
    alpha_L_tb_t=Zsym["alpha_L_tb_t"]
    alpha_L_tb_b=Zsym["alpha_L_tb_b"]


    w_min=1e-4
    Vmin_g = 1e-12  # gas cushion
    rho_o = 760.0
    rho_w = 1000.0
    D_bh = 0.2
    L_bh = 75.0
    S_bh = ca.pi * D_bh ** 2 / 4.0
    V_bh = S_bh * L_bh

    D_tb = 0.134
    L_tb = 1973.0
    S_tb = ca.pi * D_tb ** 2 / 4.0
    V_tb = S_tb * L_tb

    V_L_t = m_o_t / rho_o + m_w_t / rho_w
    V_L_b = m_o_b / rho_o + m_w_b / rho_w

    # ---------------------
    # 6) Objective: minimize ||dx||^2 (scaled)
    # ---------------------
    obj=0
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

    # enforce: V_L <= V - Vmin_g
    g_list.append(V_L_t)
    lbg.append(0.0)
    ubg.append(float(V_tb - Vmin_g))

    if is_dae:
        g_list.append(V_L_b)
        lbg.append(0.0)
        ubg.append(float(V_bh - Vmin_g))

    #
    # g_list.append(log_arg_tb_safe)
    # lbg.append(1e-12)
    # ubg.append(1e20)
    #
    # g_list.append(log_arg_bh)
    # lbg.append(1e-12)
    # ubg.append(1e20)
    #

    for a in [alpha_L_tb_b, alpha_L_tb_t,alpha_L_tb]:
        g_list.append(a)
        lbg.append(0.0)
        ubg.append(1.0)

    # for pbar in [P_an_t_bar, P_an_b_bar, P_tb_t_bar, P_tb_b_bar, P_bh_bar]:
    #     g_list.append(pbar)
    #     lbg.append(1e-6)  # >0 bar
    #     ubg.append(1e20)

    # production choke forward
    g_list.append(dP_tb_choke_bar)
    lbg.append(0)
    ubg.append(1e20)
    #
    # gas source -> annulus
    g_list.append(dP_gs_an_bar)
    lbg.append(0)
    ubg.append(1e20)

    # reservoir -> well
    g_list.append(dP_res_bh_bar)
    lbg.append(0)
    ubg.append(1e20)

    g_list.append(dP_int)
    lbg.append(0)
    ubg.append(1e20)

    # instead of dP_an_tb_bar >= 0, enforce nonnegative injection (already true)
    g_list.append(w_G_inj)
    lbg.append(w_min)
    ubg.append(1e20)

    g_list.append(w_out-w_G_inj)
    lbg.append(w_min)
    ubg.append(1e20)

    g_list.append(w_up)
    lbg.append(w_min)
    ubg.append(1e20)

    g_list.append(w_res)
    lbg.append(w_min)
    ubg.append(1e20)


    g_nlp=ca.vertcat(*g_list)

    # ---------------------
    # 8) Bounds on y
    # ---------------------
    # m_G_an=y[0]
    # m_G_t=y[1]
    # m_o_t=y[2]
    # m_w_t=y[3]
    # m_G_b=y[4]
    # m_o_b=y[5]
    # m_w_b =y[6]
    if is_dae:
        y_lb=[0.0,0.0,0.0,0,0,0,0]
        y_ub=[1e20,1e20,1e20,1e20,1e20,1e20,1e20]
    else:
        y_lb=[0.0,0.0,0.0]
        y_ub=[1e20,1e20,1e20]


    # ---------------------
    # 9) Bounds on z
    # ---------------------
    if is_dae:
        # z = [
        #   P_bh_guessed, P_bh_t_guessed, P_tb_b_guessed, P_tb_t_guessed,
        #   w_res_guessed, w_up_guessed, H_g_bh_guessed, H_g_tb_guessed
        # ]
        z_lb = [1e5, 1e5, 0.0,0.0]
        z_ub = [5e7, 5e7, 50,50]

        lbx = ca.DM(y_lb + z_lb).reshape((nx + nz, 1))
        ubx = ca.DM(y_ub + z_ub).reshape((nx + nz, 1))
    else:
        lbx = ca.DM(y_lb).reshape((nx, 1))
        ubx = ca.DM(y_ub).reshape((nx, 1))

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

    # if not bool(stats.get("success", False)):
    #     print("\n---- IPOPT FAILURE DIAGNOSTICS ----")
    #     print("Status:", stats.get("return_status", ""))
    #
    #     x_last = sol["x"]
    #
    #     if is_dae:
    #         y_last = x_last[0:nx]
    #         z_last = x_last[nx:nx + nz]
    #         dx_last, g_last, out_last = model["F_all"](y_last, z_last, u_val_dm)
    #
    #         dx_np = np.array(dx_last, dtype=float).reshape(-1)
    #         g_np = np.array(g_last, dtype=float).reshape(-1)
    #
    #         print("||dx|| =", np.linalg.norm(dx_np))
    #         print("||g||  =", np.linalg.norm(g_np))
    #         print("dx =", dx_np)
    #         print("g  =", g_np)
    #
    #     else:
    #         y_last = x_last
    #         dx_last, out_last = model["F_all"](y_last, u_val_dm)
    #
    #         dx_np = np.array(dx_last, dtype=float).reshape(-1)
    #
    #         print("||dx|| =", np.linalg.norm(dx_np))
    #         print("dx =", dx_np)
    #
    #     print("------------------------------------\n")

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


# ------------------------------------------------------------
# Build rigorous steady-state model (DAE)
# ------------------------------------------------------------




#
# ------------------------------------------------------------
# Choose a test point (u, y_guess, z_guess)
# ------------------------------------------------------------

if __name__ == "__main__":

    u = [1.00,1.00]

    surrogate=False

    if not surrogate:
        well_func = make_glc_well_rigorous( BSW=0.20, GOR=0.05, PI=3.0e-6)
        model_rigorous = build_steady_state_model(
            f_func=well_func,
            state_size=7,
            control_size=2,
            alg_size=4,
            name="glc_ss_rigorous",
            out_name=Z_NAMES
        )

        y_guess=[3679.08033973,
                289.73390193,
                3167.56224658,
                1041.96126532,
                50.46858403,
                759.52720527,
                249.84447542]

        z_guess = [8.75897957e+06,
                   8.42155186e+06,
                   2.17230613e+01,
                   2.17230613e+01]
        y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
            model=model_rigorous,
            u_val=u,
            y_guess=y_guess,
            z_guess=z_guess)
    else:
        well_func = make_glc_well_surrogate(BSW=0, GOR=0, PI=2.4e-6)
        model_surrogate = build_steady_state_model(
            f_func=well_func,
            state_size=3,
            control_size=2,
            alg_size=None,
            name="glc_ss_rigorous",
            out_name=Z_NAMES
        )

        y_guess = [50,500,500]
        u = [1.00, 1.00]

        y_star,dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
            model=model_surrogate,
            u_val=u,
            y_guess=y_guess
    )


    print("status:", stats.get("return_status", ""), "success:", stats.get("success", False))
    print("y*:", np.array(y_star).reshape(-1))
    if not surrogate:
        print("z*:", np.array(z_star).reshape(-1))
        print("||g|| :", np.linalg.norm(np.array(g_star).reshape(-1)))
    print("||dx||:", np.linalg.norm(np.array(dx_star).reshape(-1)))
    print(out_star)
    print("stable?:", stable)

    if surrogate:
    # Pretty-print the OUT vector by name (uses model["Z_names"])
        print("\n--- out* (named) ---")
        print_z_grouped(out_star, model_surrogate["Z_NAMES"])
    else:
        print("\n--- out* (named) ---")
        print_z_grouped(out_star, model_rigorous["Z_NAMES"])

