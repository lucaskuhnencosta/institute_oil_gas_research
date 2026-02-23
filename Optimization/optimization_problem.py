"""
This algorithm solves a oil and gas production optimization problem no matter the model and the variable names
"""

from Surrogate_ODE_Model.glc_01_casadi import glc_casadi
from Surrogate_ODE_Model.glc_02_bsw_casadi import glc_bsw_casadi
from Utilities.block_builders import build_steady_state_model
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict

def print_z_grouped(z_star, names, floatfmt="{: .6f}"):
    z = np.array(z_star).astype(float).reshape(-1)
    if len(z) != len(names):
        raise ValueError(
            f"z length mismatch: got {len(z)} values but {len(names)} names"
        )
    pairs = list(zip(names, z))
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
    for group_name in sorted(groups.keys()):
        print("\n" + "="*60)
        print(f"{group_name.upper()}")
        print("="*60)

        for name, val in groups[group_name]:
            print(f"{name:25s} = {floatfmt.format(val)}")


def optimize_production(
        model,   # Output of a CasADi model (steady-state) that has been assembled
        u_val,   # list/array shape (nu,)
        y_guess  # list/array shape (nx,)
        ):

    # ---------------------
    # 1) Unpack model pieces
    # ---------------------
    y_sym=model["y"] #MX nx
    u_sym=model["u"] #MX nu
    dx_expr=model["dx"] #MX nx
    z_expr=model["z"] # MX nz
    nx=model["nx"]
    nu=model["nu"]

    # ---------------------
    # 2) Decision variable (unknown equilibrium state)
    # ---------------------
    y_var=ca.MX.sym("y_var",nx)
    u_var=ca.MX.sym("u_var",nu)
    x_var=ca.vertcat(y_var,u_var) #decision vector

    # ---------------------
    # 3) Create a function from original expressions, then re-call it
    # ---------------------
    F_all=ca.Function("F_all_internal",[y_sym,u_sym],[dx_expr,z_expr])
    F_A=model["F_A"]
    dx_var,z_var=F_all(y_var,u_var) #Symbolic dx,z as functions of y_var, u_par
    # Build dictionary: name -> symbolic expression
    Zsym = {name: z_var[i] for i, name in enumerate(model["Z_names"])}

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
    alpha_G_tb_b = Zsym["alpha_G_tb_b"]
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

    w_w_out = Zsym["w_w_out"]
    w_o_out = Zsym["w_o_out"]

    # ---------------------
    # 5) Objective: maximize production
    # ---------------------
    obj=-w_o_out
    # ---------------------
    # 6) Stability boundary (polynomial fit): u2 >= b_hat(u1) + delta
    # ---------------------
    u1 = u_var[0]
    u2 = u_var[1]

    a, b, c = -0.70912576, 1.09333628, 0.14695562  # b_hat(u1)=a*u1^2+b*u1+c
    delta = 0.02  # safety margin in u2 units (tune)

    b_hat = a * u1 ** 2 + b * u1 + c
    b_hat = ca.fmin(1.0, ca.fmax(0.0, b_hat))

    g_stab = u2 - (b_hat + delta)  # must be >= 0

    # ---------------------
    # 6) Constraints
    # ---------------------

    g_list=[]
    lbg=[]
    ubg=[]

    # Enforce bottomhole pressure >= 100 bar
    g_list.append(P_bh_bar)
    lbg.append(100.0)
    ubg.append(1e20)

    g_list.append(g_stab)
    lbg.append(0.0)
    ubg.append(1e20)

    # f(u,y)=0
    g_list.append(dx_var)
    lbg.extend([0] * nx)
    ubg.extend([0] * nx)

    g_list.append(log_arg_tb_safe)
    lbg.append(1e-12)
    ubg.append(1e20)

    g_list.append(log_arg_bh)
    lbg.append(1e-12)
    ubg.append(1e20)

    # alpha_L_tb_b, alpha_L_tb_t, alpha_G_tb_t in [0,1]
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

    g=ca.vertcat(*g_list)

    # ---------------------
    # 7) Bounds on y
    # ---------------------
    y_lb=[0.0,0.0,0.0]
    y_ub=[1e20,1e20,1e20]

    u_lb=[0.0,0.0]
    u_ub=[1.0,1.0]

    lbx = ca.DM(y_lb + u_lb).reshape((nx + nu, 1))
    ubx = ca.DM(y_ub + u_ub).reshape((nx + nu, 1))

    # ---------------------
    # 8) NLP definition
    # ---------------------
    nlp = {"x": x_var, "f": obj, "g": g}
    solver = ca.nlpsol("eq_solver", "ipopt", nlp)

    # -------------------------
    # Pack numeric inputs
    # -------------------------
    y0 = ca.DM(y_guess).reshape((nx, 1))
    u0 = ca.DM(u_val).reshape((nu, 1))  # reuse u_val as initial guess for u
    x0 = ca.vertcat(y0, u0)
    # -------------------------
    # Solve
    # -------------------------
    sol = solver(
        x0=x0,
        lbx=lbx,
        ubx=ubx,
        lbg=ca.DM(lbg) if len(lbg) else ca.DM([]),
        ubg=ca.DM(ubg) if len(ubg) else ca.DM([]),
    )

    x_star = sol["x"]
    y_star = x_star[:nx]
    u_star = x_star[nx:nx + nu]

    dx_star, z_star = F_all(y_star, u_star)

    #Stability classification
    A_star = F_A(y_star, u_star)
    A_num=np.array(A_star,dtype=float)
    eig=np.linalg.eigvals(A_num)

    tol=1e-8
    stable=bool(np.all(np.real(eig)<-tol))

    stats=solver.stats()

    return y_star,u_star, dx_star, z_star,eig,stable,stats



# 1) assemble model
model = build_steady_state_model(glc_casadi,
                                 state_size=3,
                                 control_size=2,
                                 name="glc")

# 2) solve one point
u = [0.20, 0.05]
# y_guess=[3582.4731,311.7586,8523.038]
y_guess = [3919.7688, 437.16663, 7956.1206]

y_star, u_star,dx_star, z_star, eig,stable,stats = optimize_production(
    model=model,
    u_val=u,
    y_guess=y_guess,
)

print("status:", stats["return_status"], "success:", stats["success"])
print("y*:", np.array(y_star).squeeze())
print("u*:", np.array(u_star).squeeze())
print("dx*:", np.array(dx_star).squeeze())
print("||dx||:", np.linalg.norm(np.array(dx_star).squeeze()))
# print("z*:", np.array(z_star).squeeze())
print("eig:", eig)
print("stable:", stable)
print("\n--- z* (named) ---")
Z_NAMES=model["Z_names"]
print_z_grouped(z_star, Z_NAMES)  # set ncols=1 if you prefer

