from Surrogate_ODE_Model.glc_coarse_casadi import glc_casadi
from Surrogate_ODE_Model.glc_surrogate_casadi import make_glc_well_surrogate, Z_NAMES
from Rigorous_DAE_model.glc_truth_casadi import make_glc_well_rigorous, Z_NAMES
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


# ------------------------------------------------------------
# Build rigorous steady-state model (DAE)
# ------------------------------------------------------------
well_func=make_glc_well_rigorous(BSW=0,GOR=0,PI=2.4e-6)
model_rigorous = build_steady_state_model(
    f_func=well_func,
    state_size=7,
    control_size=2,
    alg_size=4,
    name="glc_ss_rigorous",
    out_name=Z_NAMES
)

#
# ------------------------------------------------------------
# Choose a test point (u, y_guess, z_guess)
# ------------------------------------------------------------
u = [0.30, 0.25]
# # y = [m_G_an, m_G_t, m_o_t, m_w_t, m_G_b, m_o_b, m_w_b]
# print(f"Initial guess is:{y_guess}")
# print(f"z is (in the guess:){z_guess}")
# P0=20e5
#
#

y_guess = [
    50.0,     # m_G_an
    500.0,     # m_G_t
    500.0,   # m_o_t
    0.0,   # m_w_t
    0.0,     # m_G_b
    1785,  # m_o_b
    0.0,   # m_w_b
]
#
# z = [P_bh, P_bh_t, P_tb_b, P_tb_t, w_res, w_up, H_g_bh, H_g_tb]
z_guess = [
    150e5,  # P_bh_guessed
    120e5,  # P_tb_b_guessed
    20.0,   # w_res_guessed
    20.0,   # w_up_guessed
]
# # z_guess = [..., P_tb_t_guessed, ...]
# z_guess[3] = float(P0 + 2e5)  # e.g. P0 + 2 bar
# z_guess[2] = z_guess[3] + 1.0e7  # crude: add ~100 bar so g5 has room

# ------------------------------------------------------------
# Solve equilibrium (dx=0, g=0)
# ------------------------------------------------------------
y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
    model=model_rigorous,
    u_val=u,
    y_guess=y_guess,
    z_guess=z_guess
)

print("status:", stats.get("return_status", ""), "success:", stats.get("success", False))
print("y*:", np.array(y_star).reshape(-1))
print("z*:", np.array(z_star).reshape(-1))
print("||dx||:", np.linalg.norm(np.array(dx_star).reshape(-1)))
print("||g|| :", np.linalg.norm(np.array(g_star).reshape(-1)))
print("stable?:", stable)

# Pretty-print the OUT vector by name (uses model["Z_names"])
print("\n--- out* (named) ---")
print_z_grouped(out_star, model_rigorous["Z_names"])


# model_surrogate = build_steady_state_model(glc_well_01_surrogate_casadi,
#                                      state_size=3,
#                                      control_size=2)
#
# # model_rigorous=build_steady_state_model(glc_well_01_rigorous_casadi,
# #                                         state_size=3,
# #                                         alg_size=3,
# #                                         control_size=2)
#
# u = [0.90, 0.90]
# y_guess = [3611.50376343,253.76732476,6696.68639922]
# z_guess = [120e5, 140e5, 10.0]  # [P_tb, P_bh, w_res] initial guesses (example)
#
# y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
#     model=model_surrogate,
#     u_val=u,
#     y_guess=y_guess,
# )
#
# #
# # y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
# #     model=model_rigorous,
# #     u_val=u,
# #     y_guess=y_guess,
# #     z_guess=z_guess
# # )
#
# print("status:", stats["return_status"], "success:", stats["success"])
# print("y*:", np.array(y_star).squeeze())
# print("dx*:", np.array(dx_star).squeeze())
# print("||dx||:", np.linalg.norm(np.array(dx_star).squeeze()))
# # print("z*:", np.array(z_star).squeeze())
# # print("\n--- z* (named) ---")
# # Z_NAMES=model_surrogate["Z_names"]
# # # print_z_grouped(out_star, Z_NAMES)  # set ncols=1 if you prefer
# #
#
#
# import numpy as np
# #
# # def make_initial_guess_oil_filled(
# #     *,
# #     V_bh, V_tb, V_an,         # volumes [m^3]
# #     rho_o, rho_w,             # densities [kg/m^3]
#     P0, P_gs,                 # outlet and gas-supply pressures [Pa]
#     P_res=None,               # reservoir pressure [Pa] (optional)
#     BSW=0.0, GOR=0.0,          # for this initializer we assume ~0 anyway
#     # heuristic pressure drops:
#     dp_choke=5e5,             # Pa (5 bar) ensure outflow not dead
#     dp_tb=8e6,                # Pa (80 bar) TB hydro+friction rough
#     dp_bh=5e5,                # Pa (5 bar) BH segment rough
#     dp_an=2e6,                # Pa (20 bar) annulus hydro rough
#     w_guess=10.0,             # kg/s initial guess for w_res and w_up
#     H_g_tb_guess=0.7,         # encourage gas in TB for venting injection
#     H_g_bh_guess=0.0
# ):
#     """
#     Returns y_guess (7,) and z_guess (8,) for your rigorous DAE model.
#
#     y = [m_G_an, m_G_t, m_o_t, m_w_t, m_G_b, m_o_b, m_w_b]
#     z = [P_bh, P_bh_t, P_tb_b, P_tb_t, w_res, w_up, H_g_bh, H_g_tb]
#     """
#
#     # -----------------------------
#     # States: fill BH & TB with oil
#     # -----------------------------
#     m_o_b = rho_o * V_bh
#     m_w_b = 0.0
#     m_G_b = 0.0
#
#     m_o_t = rho_o * V_tb
#     m_w_t = 0.0
#     m_G_t = 0.0
#
#     # -----------------------------
#     # Pressures: build a ladder
#     # -----------------------------
#     P_tb_t = float(P0 + dp_choke)          # keep choke ΔP positive
#     P_tb_b = float(P_tb_t + dp_tb)
#     P_bh_t = float(P_tb_b + 1e3)           # interface closure near-zero
#     P_bh   = float(P_bh_t + dp_bh)
#
#     # Annulus pressures: start near supply
#     # If you want annulus top ≈ P_gs, keep P_an_t≈P_gs - small
#     # but model defines P_an_t from m_G_an, so we set m_G_an accordingly later.
#     # Here we only need a consistent guessed annulus bottom pressure for injector ΔP.
#     P_an_t_target = float(max(P_gs - 1e5, 1e5))     # 1 bar below supply
#     P_an_b_target = float(P_an_t_target + dp_an)
#
#     # -----------------------------
#     # Annulus gas mass (state m_G_an)
#     # Using ideal gas: P_an_t = R*T_an*m_G_an/(M_G*V_an)
#     # We do NOT know (R, T_an, M_G) here, so:
#     # - either you pass a function to compute it,
#     # - or you just guess m_G_an and let IPOPT adjust.
#     # Practical: set a moderate gas mass; IPOPT will fix it.
#     # -----------------------------
#     m_G_an = 1000.0  # kg (rough). If too high/low, adjust once and forget.
#
#     # -----------------------------
#     # Algebraic guesses
#     # -----------------------------
#     w_res = float(w_guess)
#     w_up  = float(w_guess)
#
#     z_guess = np.array([
#         P_bh, P_bh_t, P_tb_b, P_tb_t,
#         w_res, w_up,
#         float(H_g_bh_guess),
#         float(H_g_tb_guess)
#     ], dtype=float)
#
#     y_guess = np.array([
#         float(m_G_an),
#         float(m_G_t),
#         float(m_o_t),
#         float(m_w_t),
#         float(m_G_b),
#         float(m_o_b),
#         float(m_w_b)
#     ], dtype=float)
#
#     return y_guess.tolist(), z_guess.tolist()
#
#
#
# # Well properties
# BSW = 0
# GOR = 0 # is the gas oil ratio
# PI = 3.00e-6  # is the productivity index in kg/(s.Pa)
#
# # Constants (general)
# R = 8.314  # J/(K*mol) is the universal gas constant
# g = 9.81  # m/s^2 is the gravity
# mu_o = 3.64e-3  # Pa.s is the viscosity
#
# # Constants (fluid)
# mu_w = 1.00e-3
# rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
# rho_w = 1000
# rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
# mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
# M_G = 0.0167  # (kg/mol) is the gas molecular weight
#
# # Temperatures
# T_an = 348  # K is the annulus temperature
# T_tb = 369.4  # K is the tubing temperature
# T_bh=371.5
#
# # Volumes, lengths and areas
# ### Annulus ###
# V_an = 64.34  # m^3 is the annulus volume
# L_an = 2048  # m is the length of the annulus
#
# ### Tubing bottom
# L_bh = 75  # m is the length below the injection point
# S_bh = 0.0314  # m^2 is the cross section below the injection point
# D_bh = 2 * np.sqrt(S_bh / np.pi) # 0.2 m diameter
# V_bh = S_bh * L_bh
#
# ### Tubing top
# L_tb = 1973
# D_tb = 0.134  # 0.13m diameter
# S_tb=(np.pi*D_tb**2)/4
# V_tb=S_tb*L_tb
#
# # Pressures
# P_gs = 140e5  # 140bar is the gas source pressure
# P_res = 160e5  # 160bar, the constant reservoir pressure
# P_0 = 20e5  # pressure downstream of choke
#
# # Chokes
# K_gs = 9.98e-5  # is the gas lift choke constant
# K_inj = 1.40e-4  # is the injection valve choke constant
# K_pr = 2.90e-3  # is the production choke constant
# K0_int=0.01
#
# # Friction
# epsilon_tubing = 3e-4
#
# # Slip model
# C_0_b=1.15
# C_0_t=1.20
#
# V_d_b=0.25
# V_d_t=0.40
#
#
# y_guess, z_guess = make_initial_guess_oil_filled(
#     V_bh=V_bh, V_tb=V_tb, V_an=V_an,
#     rho_o=rho_o, rho_w=rho_w,
#     P0=P_0, P_gs=P_gs,
#     dp_choke=5e5,   # 5 bar
#     dp_tb=8e6,      # 80 bar
#     dp_bh=5e5,      # 5 bar
#     dp_an=2e6,      # 20 bar
#     w_guess=10.0,
#     H_g_tb_guess=0.7
# )