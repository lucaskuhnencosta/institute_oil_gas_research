import casadi as ca
from application.simulation_engine import make_model
from utilities.block_builders import *
from networks.networks import *


def optimize_single_well_production_NN(
        F_u2z,
        F_pinn,
        u_guess=(0.5,0.5),
        P_max_tb_b_bar=120,
        P_min_bh_bar=90,
        ):
    ipopt_opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 6000,
        "ipopt.tol": 1e-10,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    }

    # ---------------------
    # Decision variable (single well)
    # ---------------------
    u = ca.MX.sym("u", 2)  # [u1,u2]

    u1 = u[0]
    u2 = u[1]

    # Evaluate NN surrogate
    z = F_u2z(u=u)["z"] if isinstance(F_u2z(u=u), dict) else F_u2z(u)  # robust
    m_o_out = z[0]
    P_bh = z[1]
    P_tb_b = z[2]

    # ---------------------
    # Objective: maximize oil
    # ---------------------
    obj = -m_o_out

    # ---------------------
    # Constraints
    #   P_bh >= P_min
    #   P_tb_b <= P_max
    # ---------------------
    b_hat = -0.3268*u1*u1 + 0.5116*u1 + 0.01914
    g_stab = u2 - b_hat


    g = ca.vertcat(P_bh, P_tb_b,g_stab)
    lbg = ca.DM([float(P_min_bh_bar), -ca.inf,0.0])
    ubg = ca.DM([ca.inf, float(P_max_tb_b_bar),ca.inf])

    # ---------------------
    # Bounds / initial guess
    # ---------------------
    lbx = ca.DM([0.05, 0.10])
    ubx = ca.DM([1.0, 1.0])
    x0 = ca.DM(list(u_guess))

    # ---------------------
    # Solve NLP
    # ---------------------
    nlp = {"x": u, "f": obj, "g": g}
    solver = ca.nlpsol("single_well_solver", "ipopt", nlp, ipopt_opts)

    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    u_star = ca.DM(sol["x"]).full().flatten()
    y_star = ca.DM(F_pinn(u=u_star)["y"]).full().flatten()
    z_star = ca.DM(F_u2z(u=u_star)["z"] if isinstance(F_u2z(u=u_star), dict) else F_u2z(u_star)).full().flatten()

    return {
        "stats": stats,
        "u_star": u_star,  # [u1*, u2*]
        "y_star": y_star,
        "z_star": z_star,  # [m_o_out*, P_bh*, P_tb_b*]
        "m_o_out": float(z_star[0]),
        "P_bh": float(z_star[1]),
        "P_tb_b": float(z_star[2]),
    }

pinn = PINN(
    hidden_units=[64,64,64]
)

algnn = AlgNN(
    hidden_units=[64,64,64,64]
)


# 1) Load your torch models
# from your_code import PINN, AlgNN, load_model_weights
pinn = load_model_weights(pinn, "../training/PINN.pth")
algnn  = load_model_weights(algnn, "../training/AlgNN.pth")

# 2) Extract weights
pinn_w = extract_pinn_standard_weights(pinn)
alg_w  = extract_algnn_standard_weights(algnn)

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
    alg_y_min=alg_y_min,   alg_y_max=alg_y_max,
    alg_u_min=alg_u_min,   alg_u_max=alg_u_max,
    alg_z_min=alg_z_min,   alg_z_max=alg_z_max,
)

# F_u2z is your chained CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
res = optimize_single_well_production_NN(
    F_u2z=F_u2z,
    F_pinn=F_pinn,
    u_guess=(0.8, 0.8),
    P_min_bh_bar=90.0,
    P_max_tb_b_bar=120.0,
)

print("Solver success:", res["stats"]["success"])
print("u* =", res["u_star"])
print("y* =", res["y_star"])
print("m_o_out* =", res["m_o_out"])
print("P_bh* =", res["P_bh"])
print("P_tb_b* =", res["P_tb_b"])

from solvers.steady_state_solver import solve_equilibrium_ipopt
u1=res["u_star"][0]
u2=res["u_star"][1]

y_guess=res["y_star"]

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

print('\n\n')
print(f"Now we take this control and apply to the plant...")
model=make_model("rigorous",BSW=0.20,GOR=0.05,PI=3.0e-6)
y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
    model=model,
    u_val=[u1, u2],
    y_guess=y_guess_rig,
    z_guess=z_guess_rig
)
print(f"y_star of the model used to train this NN at u*={y_star}")
print(f"And the pressure bottomhole is p_bh={out_star[15]} and w_out={out_star[38]}")