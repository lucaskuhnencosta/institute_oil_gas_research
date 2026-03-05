import casadi as ca
import numpy as np

from Optimization.production_optimizer_NN import y_guess_rig, z_guess_rig

np.set_printoptions(suppress=True, precision=6)
from Networks.networks import PINN, AlgNN
from Utilities.block_builders import *
from Solvers.solve_glc_ode_equilibrium import solve_equilibrium_ipopt
from Rigorous_DAE_model.glc_truth_casadi import make_glc_well_rigorous, Z_NAMES_RIG
from Optimization.api_interface_simulator import plant_zeroth_and_first_order

#####################################################################
# 0 - ALL FUNCTIONS DEFINITIONS
#####################################################################

def build_rk_derivatives(F_u2z):
    u = ca.MX.sym("u",2)
    y = F_u2z(u=u)["z"]   # r_k(u)
    J = ca.jacobian(y,u)  # ∂y/∂u  (3x2)
    F_rk = ca.Function(
        "F_rk",
        [u],
        [y,J],
        ["u"],
        ["z","J"]
    )
    return F_rk

def print_rk_information(F_rk, u_k):

    res = F_rk(u=u_k)

    yk = np.array(res["z"]).reshape((-1,))
    Jk = np.array(res["J"])

    print("\n========== SURROGATE LOCAL MODEL ==========")
    print(f"u_k = {np.array(u_k)}")

    print("\nr_k(u_k)  (0th order term)")
    print(yk)

    print("\n∇r_k(u_k)  (Jacobian dy/du)")
    print(Jk)

    print("\nInterpretation:")
    print("Rows  = states [y1,y2,y3]")
    print("Cols  = controls [u1,u2]")
    print("Entry (i,j) = ∂y_i / ∂u_j")
    print("===========================================\n")

def build_corrected_surrogate(F_u2z,
                              u_k,
                              z0_plant,
                              J_plant):
    """
       Builds corrected model:
           z_corr(u) = z_surr(u) + c_k + C_k (u - u_k)
       where:
           c_k = z0_plant - z_surr(u_k)
           C_k = J_plant - J_surr(u_k)

       Inputs:
         F_u2z: CasADi Function  u -> z (3x1)
         u_k:   iterable length 2
         z0_plant: np array shape (3,) with plant outputs at u_k
         J_plant:  np array shape (3,2) with plant dz/du at u_k

       Returns:
         F_u2z_corr: CasADi Function u -> z_corr
         plus dict with (c_k, C_k, z0_surr, J_surr)
   """
    u_k = np.array(u_k, dtype=float).reshape((2,))
    z0_plant = np.array(z0_plant, dtype=float).reshape((3,))
    J_plant = np.array(J_plant, dtype=float).reshape((3, 2))

    # --- symbolic u ---
    u = ca.MX.sym("u", 2)

    # base surrogate
    z_surr = F_u2z(u=u)["z"]  # (3x1)

    # evaluate surrogate at u_k and its Jacobian
    u_sym = ca.MX.sym("u_sym", 2)
    z_sym = F_u2z(u=u_sym)["z"]
    J_sym = ca.jacobian(z_sym, u_sym)
    F_zJ = ca.Function("F_zJ", [u_sym], [z_sym, J_sym], ["u"], ["z", "J"])

    z0_surr = np.array(F_zJ(u=u_k)["z"]).reshape((3,))
    J_surr = np.array(F_zJ(u=u_k)["J"]).reshape((3, 2))

    # corrections
    c_k = z0_plant - z0_surr  # (3,)
    C_k = J_plant - J_surr  # (3,2)

    # build corrected expression
    du = u - ca.DM(u_k)
    z_corr = z_surr + ca.DM(c_k) + ca.DM(C_k) @ du

    # output as function (same naming convention as your other ones)
    F_u2z_corr = ca.Function(
        "F_u2z_corr",
        [u],
        [z_corr],
        ["u"],
        ["z"]
    )

    info = {
        "u_k": u_k,
        "z0_surr": z0_surr,
        "J_surr": J_surr,
        "z0_plant": z0_plant,
        "J_plant": J_plant,
        "c_k": c_k,
        "C_k": C_k,
    }

    return F_u2z_corr, info



def solve_trust_region_subproblem(
        F_u2z,
        u_k,
        Delta,
        u_min,
        u_max,
        P_min_bh,
        P_max_tb_b,
        ipopt_opts=None,
):
    u_k=np.array(u_k,dtype=float).reshape((2,))


    # Decision variable (DOFs only)
    u=ca.MX.sym("u",2)

    z=F_u2z(u=u)["z"]
    m_o=z[0]
    P_bh=z[1]
    P_tb=z[2]

    g=[]
    lbg=[]
    ubg=[]

    # Inequality constraints
    g += [P_bh];      lbg += [P_min_bh];  ubg += [ca.inf]
    g += [P_tb];      lbg += [-ca.inf];   ubg += [P_max_tb_b]

    # Stability constraint u2 >= b_hat(u1)
    u1 = u[0]; u2 = u[1]
    b_hat = -0.3268*u1*u1 + 0.5116*u1 + 0.01914
    g += [u2 - b_hat]; lbg += [0.0]; ubg += [ca.inf]

    du=u-ca.DM(u_k)
    g+=[ca.dot(du,du)]
    lbg+=[-ca.inf]
    ubg+=[Delta**2]

    # Objective (IPOPT minimizes)
    f = -m_o

    nlp = {"x": u, "f": f, "g": ca.vertcat(*g)}
    if ipopt_opts is None:
        ipopt_opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-10,
            "ipopt.constr_viol_tol": 1e-8,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.linear_solver": "mumps",
        }

    solver = ca.nlpsol("tr_solver", "ipopt", nlp, ipopt_opts)

    sol = solver(
        x0=ca.DM(u_k),
        lbx=ca.DM(u_min),
        ubx=ca.DM(u_max),
        lbg=ca.DM(lbg),
        ubg=ca.DM(ubg),
    )

    u_star = np.array(sol["x"]).reshape((2,))
    z_star = np.array(F_u2z(u=u_star)["z"]).reshape((-1,))

    return {
        "success": bool(solver.stats().get("success", False)),
        "stats": solver.stats(),
        "u_star": u_star,
        "m_o_out": float(z_star[0]),
        "P_bh": float(z_star[1]),
        "P_tb_b": float(z_star[2]),
    }



#####################################################################
# 1 - SOLVER INITIALIZATION. RUNS JUST ONCE
#####################################################################

#--------------------------------------------------------------------
# 1.1 - OPTIMIZATION SETTINGS
#--------------------------------------------------------------------

u_min=[0.05,0.10]
u_max=[1.00,1.00]
umin = np.array(u_min, dtype=float).reshape((2,))
umax = np.array(u_max, dtype=float).reshape((2,))
P_min_bh=90
P_max_tb_b=120

#--------------------------------------------------------------------
# 1.2 - Z_NAMES AND REAL PLANT
#--------------------------------------------------------------------

Z_NAMES = Z_NAMES_RIG
well_func = make_glc_well_rigorous(BSW=0.20, GOR=0.05, PI=3.0e-6)
plant_model = build_steady_state_model(
    f_func=well_func,
    state_size=7,
    control_size=2,
    alg_size=4,
    name="glc_ss_rigorous",
    out_name=Z_NAMES
)

#--------------------------------------------------------------------
# 1.3 - SURROGATE MODEL DEFINITIONS
#--------------------------------------------------------------------

pinn = PINN(
        hidden_units=[64, 64, 64])
algnn = AlgNN(
    hidden_units=[64, 64, 64, 64])

# 1) Load your torch models
# from your_code import PINN, AlgNN, load_model_weights
pinn = load_model_weights(pinn, "../Training/PINN.pth")
algnn = load_model_weights(algnn, "../Training/AlgNN.pth")

# 2) Extract weights
pinn_w = extract_pinn_standard_weights(pinn)
alg_w = extract_algnn_standard_weights(algnn)

# 3) IMPORTANT: use the SAME scaling constants as the trained models
# You can read them from buffers:
pinn_y_min = pinn.y_min.cpu().numpy().tolist()
pinn_y_max = pinn.y_max.cpu().numpy().tolist()
pinn_u_min = pinn.u_min.cpu().numpy().tolist()
pinn_u_max = pinn.u_max.cpu().numpy().tolist()

alg_y_min = algnn.y_min.cpu().numpy().tolist()
alg_y_max = algnn.y_max.cpu().numpy().tolist()
alg_u_min = algnn.u_min.cpu().numpy().tolist()
alg_u_max = algnn.u_max.cpu().numpy().tolist()
alg_z_min = algnn.z_min.cpu().numpy().tolist()
alg_z_max = algnn.z_max.cpu().numpy().tolist()

F_pinn = build_casadi_pinn_function_standard(pinn_w,
                                             pinn_y_min,
                                             pinn_y_max,
                                             pinn_u_min,
                                             pinn_u_max)

F_u2z = build_casadi_surrogate_u2z(
    pinn_weights=pinn_w,
    algnn_weights=alg_w,
    pinn_y_min=pinn_y_min, pinn_y_max=pinn_y_max,
    pinn_u_min=pinn_u_min, pinn_u_max=pinn_u_max,
    alg_y_min=alg_y_min, alg_y_max=alg_y_max,
    alg_u_min=alg_u_min, alg_u_max=alg_u_max,
    alg_z_min=alg_z_min, alg_z_max=alg_z_max,
)
F_rk = build_rk_derivatives(F_u2z)

#--------------------------------------------------------------------
# 1.4 - INITIAL GUESSES
#--------------------------------------------------------------------

y_guess_rig_init = [3679.08033973,
                    289.73390193,
                    3167.56224658,
                    1041.96126532,
                    50.46858403,
                    759.52720527,
                    249.84447542]

z_guess_rig_init = [8.75897957e+06,
                    8.42155186e+06,
                    2.17230613e+01,
                    2.17230613e+01]

u_k_init=[0.5,0.5]
Delta_init=0.05


###############################################################
# 2 - BUILD THE MODEL R_K
###############################################################

u_k=u_k_init
y_guess_rig=y_guess_rig_init
z_guess_rig=z_guess_rig_init
Delta=Delta_init

####



simulation_import_api=plant_zeroth_and_first_order(
    model=plant_model,
    u_k=u_k,
    y_guess=y_guess_rig,
    z_guess=z_guess_rig
)


print_rk_information(F_rk,u_k)

plant_z0=simulation_import_api["t0"]
plant_J=simulation_import_api["Jt"]

F_u2z_corr, corr_info = build_corrected_surrogate(
    F_u2z=F_u2z,
    u_k=u_k,
    z0_plant=plant_z0,
    J_plant=plant_J
)

print("\n==== CORRECTION TERMS ====")
print("u_k =", corr_info["u_k"])
print("z0_surr  =", corr_info["z0_surr"])
print("z0_plant =", corr_info["z0_plant"])
print("c_k      =", corr_info["c_k"])
print("\nJ_surr =\n", corr_info["J_surr"])
print("\nJ_plant=\n", corr_info["J_plant"])
print("\nC_k =\n", corr_info["C_k"])
print("==========================\n")



###############################################################
# Here we assemble and solve the TRSP
###############################################################
# F_u2z is your chained CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
res = solve_trust_region_subproblem(
    F_u2z=F_u2z_corr,
    u_k=u_k,
    Delta=Delta,
    u_min=u_min,
    u_max=u_max,
    P_min_bh=P_min_bh,
    P_max_tb_b=P_max_tb_b
)

print("Solver success:", res["stats"]["success"])
print("u* =", res["u_star"])
print("m_o_out* =", res["m_o_out"])
print("P_bh* =", res["P_bh"])
print("P_tb_b* =", res["P_tb_b"])

u_k=np.array(u_k,dtype=float).reshape((2,))
u_star=np.array(res["u_star"],dtype=float).reshape((2,))

z_k=np.array(F_u2z(u=u_k)["z"]).reshape((-1,))

m_os, Pbh_s, Ptb_s = res["m_o_out"], res["P_bh"], res["P_tb_b"]
print("m_os =", m_os)
print("Pbh_s =", Pbh_s)
print("Ptb_s =", Ptb_s)

m_ok,Pbh_k,Ptb_k=z_k[0],z_k[1],z_k[2]

#So what was the step size?
step = u_star - u_k
step_norm = float(np.linalg.norm(step))
hit_TR = step_norm >= 0.999 * float(Delta)

# Constraint slacks (>=0 means satisfied, =0 active)
slack_Pbh = float(Pbh_s - P_min_bh)  # >= 0
slack_Ptb = float(P_max_tb_b - Ptb_s)  # >= 0

u1, u2 = float(u_star[0]), float(u_star[1])
b_hat = -0.3268 * u1 * u1 + 0.5116 * u1 + 0.01914
slack_stab = float(u2 - b_hat)  # >= 0

# Improvement
dm = float(m_os - m_ok)

print("\n================ TRUST-REGION SUBPROBLEM ================")
print("success:", res["success"], res["stats"].get("return_status", ""))
print(f"u_k    = {u_k}")
print(f"u_star = {u_star}")
print(f"step   = {step} | ||step||={step_norm:.6f}  (Delta={Delta})  hit_TR={hit_TR}")

print("\nObjective:")
print(f"m_o_out(u_k)    = {m_ok:.6f}")
print(f"m_o_out(u_star) = {m_os:.6f}")
print(f"Δm_o_out        = {dm:.6f}")

print("\nConstraints at u_star (slack >= 0 means satisfied):")
print(f"P_bh     = {Pbh_s:.6f}  (>= {P_min_bh})  slack = {slack_Pbh:.6f}")
print(f"P_tb_b   = {Ptb_s:.6f}  (<= {P_max_tb_b})  slack = {slack_Ptb:.6f}")
print(f"stability u2-b_hat = {slack_stab:.6}  (>= 0)")
print("=========================================================\n")

#
# u1 = res["u_star"][0]
# u2 = res["u_star"][1]
# y_guess = res["y_star"]
#
# print('\n\n')
# print(f"Now we take this control and apply to the plant...")
# model = make_model("surrogate", BSW=0.20, GOR=0.05, PI=3.0e-6)
# y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
#     model=model,
#     u_val=[u1, u2],
#     y_guess=y_guess
# )
# print(f"y_star of the model used to train this NN at u*={y_star}")
# print(f"And the pressure bottomhole is p_bh={out_star[15]} and w_out={out_star[38]}")