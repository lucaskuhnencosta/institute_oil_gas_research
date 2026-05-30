import numpy as np
import casadi as ca
import pickle

# Step change in production choke opening u1
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from application.simulation_engine import make_model
from solvers.steady_state_solver import solve_equilibrium_ipopt
from configuration.wells import get_wells
from simulators.Z_NAMES import Z_NAMES
wells = get_wells()

def make_output_dict(out, Z_NAMES):
    """
    Convert CasADi output vector into a Python dict.
    """
    out_np = np.array(out, dtype=float).reshape(-1)
    return {name: out_np[i] for i, name in enumerate(Z_NAMES)}


def build_dae_backward_euler_step_solver(
        model,
        dt,
        solver_name="dae_be_step"
):
    """
    Build one backward-Euler DAE step solver.

    Semi-explicit DAE:

        dy/dt = f(y,z,u)
        0     = g(y,z,u)

    Backward Euler:

        y_next - y_prev - dt*f(y_next,z_next,u_next) = 0
        g(y_next,z_next,u_next) = 0

    Decision variables:

        x = [y_next; z_next]

    Parameters:

        p = [y_prev; u_next]
    """

    nx = model["nx"]
    nu = model["nu"]
    is_dae=model["is_dae"]

    # Decision variables
    y_next = ca.MX.sym("y_next", nx)
    # Parameters
    y_prev = ca.MX.sym("y_prev", nx)
    u_next = ca.MX.sym("u_next", nu)

    if is_dae:
        nz = model["nz"]
        z_next = ca.MX.sym("z_next", nz)
        x = ca.vertcat(y_next, z_next)
    else:
        nz=0
        x=y_next

    p = ca.vertcat(y_prev, u_next)

    # Model evaluation at next time
    if is_dae:
        dx_next, g_next, out_next = model["F_all"](y_next, z_next, u_next)
    else:
        dx_next, out_next = model["F_all"](y_next, u_next)
    Zsym = {name: out_next[i] for i, name in enumerate(Z_NAMES)}

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


    # Backward Euler residual
    r_dyn = y_next - y_prev - dt * dx_next

    g_list = [r_dyn]
    lbg = [0.0] * nx
    ubg = [0.0] * nx

    if is_dae:
        r_alg = g_next
        g_list.append(r_alg)
        lbg += [0.0] * nz
        ubg += [0.0] * nz

    Vmin_g = 1e-4
    V_tb = 24.99334640648408
    V_bh = 2.356194490192345
    rho_o = 760.0
    rho_w = 1000.0

    # y = [
    #   m_G_an, m_G_t, m_o_t, m_w_t,
    #   m_G_b,  m_o_b, m_w_b
    # ]

    if is_dae:
        m_o_t = y_next[2]
        m_w_t = y_next[3]
        m_o_b = y_next[5]
        m_w_b = y_next[6]

        V_L_t = m_o_t / rho_o + m_w_t / rho_w
        V_L_b = m_o_b / rho_o + m_w_b / rho_w

        g_list.append(V_L_t)
        lbg.append(0.0)
        ubg.append(float(V_tb - Vmin_g))

        g_list.append(V_L_b)
        lbg.append(0.0)
        ubg.append(float(V_bh - Vmin_g))

    for a in [alpha_L_tb_b, alpha_L_tb_t,alpha_L_tb]:
        g_list.append(a)
        lbg.append(0.0)
        ubg.append(1.0)

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
    lbg.append(Vmin_g)
    ubg.append(1e20)

    g_list.append(w_out-w_G_inj)
    lbg.append(Vmin_g)
    ubg.append(1e20)

    g_list.append(w_up)
    lbg.append(Vmin_g)
    ubg.append(1e20)

    g_list.append(w_res)
    lbg.append(Vmin_g)
    ubg.append(1e20)

    g_nlp = ca.vertcat(*g_list)

    # Feasibility problem
    obj = 0.0

    nlp = {
        "x": x,
        "p": p,
        "f": obj,
        "g": g_nlp,
    }

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.tol": 1e-10,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_constr_viol_tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
    }

    solver = ca.nlpsol(solver_name, "ipopt", nlp, opts)

    # Default bounds
    y_lb = [0.0] * nx
    y_ub = [1e20] * nx
    if is_dae:
        z_lb = [1e5, 1e5, 0.0, 0.0]
        z_ub = [5e7, 5e7, 50.0, 50.0]
        lbx = ca.DM(y_lb + z_lb)
        ubx = ca.DM(y_ub + z_ub)
    else:
        lbx = ca.DM(y_lb)
        ubx = ca.DM(y_ub)

    return solver, lbx, ubx, ca.DM(lbg), ca.DM(ubg)


def simulate_dae_backward_euler(
        model,
        y0,
        z0,
        u_fun,
        t_grid,
        Z_NAMES=None,
):
    """
    Simulate the DAE model using backward Euler.

    Parameters
    ----------
    model:
        Dictionary returned by build_steady_state_model(...).

    y0:
        Initial differential state, preferably from solve_equilibrium_ipopt.

    z0:
        Initial algebraic state, preferably from solve_equilibrium_ipopt.

    u_fun:
        Function of time: u = u_fun(t)

    t_grid:
        1D array of simulation times.

    Z_NAMES:
        Optional list of output names.

    Returns
    -------
    result:
        Dictionary with time, y, z, u, dx, g, out, and named outputs.
    """

    is_dae = bool(model["is_dae"])

    nx = model["nx"]
    nu = model["nu"]

    if is_dae:
        nz = model["nz"]
        if z0 is None:
            raise ValueError("DAE simulation requires z0.")
    else:
        nz = 0

    idx_P_bh = Z_NAMES.index("P_bh_bar")
    idx_w_o_out = Z_NAMES.index("w_o_out")
    idx_w_G_inj = Z_NAMES.index("w_G_inj")

    t_grid = np.array(t_grid, dtype=float).reshape(-1)
    nt = len(t_grid)


    y0 = np.array(y0, dtype=float).reshape(nx)

    if is_dae:
        z0 = np.array(z0, dtype=float).reshape(nz)

    # Storage
    Y = np.zeros((nt, nx))
    U = np.zeros((nt, nu))
    DX = np.zeros((nt, nx))

    if is_dae:
        Z = np.zeros((nt, nz))
        G = np.zeros((nt, nz))
    else:
        Z=None
        G=None

    OUT = []
    OUT_DICT = []

    Y[0, :] = y0
    U[0, :] = np.array(u_fun(t_grid[0]), dtype=float).reshape(nu)

    if is_dae:
        Z[0,:]=z0

    # Evaluate initial point
    if is_dae:
        dx0, g0, out0 = model["F_all"](
            ca.DM(Y[0, :]),
            ca.DM(Z[0, :]),
            ca.DM(U[0, :])
        )
        G[0, :] = np.array(g0, dtype=float).reshape(nz)
    else:
        dx0, out0 = model["F_all"](
            ca.DM(Y[0, :]),
            ca.DM(U[0, :]))

    DX[0, :] = np.array(dx0, dtype=float).reshape(nx)


    out0_np = np.array(out0, dtype=float).reshape(-1)
    OUT.append(out0_np)
    OUT_DICT.append(make_output_dict(out0, Z_NAMES))

    # Time stepping
    for k in range(nt - 1):
        t_prev = t_grid[k]
        t_next = t_grid[k + 1]
        dt = float(t_next - t_prev)

        u_next = np.array(u_fun(t_next), dtype=float).reshape(nu)

        solver, lbx, ubx, lbg, ubg = build_dae_backward_euler_step_solver(
            model=model,
            dt=dt,
            solver_name=f"dae_be_step_{k}"
        )

        # Warm start with previous solution
        if is_dae:
            x0 = ca.DM(np.concatenate([Y[k, :], Z[k, :]]))
        else:
            x0 = ca.DM(Y[k, :])

        p = ca.DM(np.concatenate([Y[k, :], u_next]))

        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p,
        )
        x_next = np.array(sol["x"], dtype=float).reshape(-1)
        y_next = x_next[:nx]
        z_next = x_next[nx:nx + nz]

        Y[k + 1, :] = y_next
        U[k + 1, :] = u_next

        if is_dae:
            Z[k + 1, :] = z_next

        if is_dae:
            dx_next, g_next, out_next = model["F_all"](
                ca.DM(y_next),
                ca.DM(z_next),
                ca.DM(u_next)
            )
            G[k + 1, :] = np.array(g_next, dtype=float).reshape(nz)
        else:
            dx_next, out_next = model["F_all"](
                ca.DM(y_next),
                ca.DM(u_next)
            )


        DX[k + 1, :] = np.array(dx_next, dtype=float).reshape(nx)
        out_next_np = np.array(out_next, dtype=float).reshape(-1)
        OUT.append(out_next_np)
        OUT_DICT.append(make_output_dict(out_next, Z_NAMES))


        if (k + 1) % 100 == 0:
            print(
                f"k = {k + 1:5d} | "
                f"t = {t_next:10.2f} s | "
                f"u = [{u_next[0]:.3f}, {u_next[1]:.3f}] | "
                f"P_bh = {out_next_np[idx_P_bh]:12.6f} bar | "
                f"w_o_out = {out_next_np[idx_w_o_out]:12.6f} kg/s | "
                f"w_G_inj = {out_next_np[idx_w_G_inj]:12.6f} kg/s"
            )
    OUT = np.vstack(OUT)

    return {
        "t": t_grid,
        "Y": Y,
        "Z": Z,
        "U": U,
        "DX": DX,
        "G": G,
        "OUT": OUT,
        "OUT_DICT": OUT_DICT,
        "Z_NAMES": Z_NAMES,
    }


well="P6"
params=wells[well]
BSW = params["BSW"]
GOR = params["GOR"]
PI = params["PI"]
K_gs = params["K_gs"]
K_inj = params["K_inj"]
K_pr = params["K_pr"]
y_guess_rig = params["y_guess_rig"]
z_guess_rig = params["z_guess_rig"]
y_guess_sur=params["y_guess_sur"]

y_ss=y_guess_rig
z_ss=z_guess_rig
y_sur=y_guess_sur

def u_fun(t):
    if t < 500.0:
        return np.array([0.80, 1.0])
    if t < 3600+500:
        return np.array([0.80, 0.80])
    elif t < 3600.0*2.0+500.0:
        return np.array([0.80, 0.60])
    elif t < 3600.0*3.0+500.0:
        return np.array([0.80, 0.40])
    elif t < 3600.0*4.0+500.0:
        return np.array([0.80, 0.20])
    else:
        return np.array([0.80, 0.20])

dt=1
t_grid = np.arange(0.0, 3600.0*4.0+500+dt, dt)


#######################################################
# THIS IS TO RUN RIGOROUS SIMULATOR
#######################################################

# model_rig = make_model("rigorous",
#                    BSW=BSW,
#                    GOR=GOR,
#                    PI=PI,
#                    K_gs=K_gs,
#                    K_inj=K_inj,
#                    K_pr=K_pr)
#
# y_ss=y_guess_rig
# z_ss=z_guess_rig
# y_sur=y_guess_sur
#
# sim_rig = simulate_dae_backward_euler(
#     model=model_rig,
#     y0=y_ss,
#     z0=z_ss,
#     u_fun=u_fun,
#     t_grid=t_grid,
#     Z_NAMES=Z_NAMES
# )
# with open("sim_rig_dt_1s.pkl", "wb") as f:
#     pickle.dump(sim_rig, f)


#######################################################
# THIS IS TO RUN SURROGATE SIMULATOR
#######################################################
model_sur = make_model("surrogate",
                   BSW=BSW,
                   GOR=GOR,
                   PI=PI,
                   K_gs=K_gs,
                   K_inj=K_inj,
                   K_pr=K_pr)



sim_sur = simulate_dae_backward_euler(
    model=model_sur,
    y0=y_sur,
    z0=None,
    u_fun=u_fun,
    t_grid=t_grid,
    Z_NAMES=Z_NAMES
)
with open("sim_sur_dt_1s.pkl", "wb") as f:
    pickle.dump(sim_sur, f)


#####################################################################
#####################################################################
#####################################################################


with open("sim_rig_dt_1s.pkl", "rb") as f:
    sim_rig = pickle.load(f)
with open("sim_sur_dt_1s.pkl", "rb") as f:
    sim_sur = pickle.load(f)

t = sim_sur["t"]
names = sim_rig["Z_NAMES"]
idx_wout = names.index("w_o_out")
idx_wginj = names.index("w_G_inj")


out_rig = sim_rig["OUT"]
out_sur=sim_sur["OUT"]


# ============================================================
# Extract data
# ============================================================

t_rig = sim_sur["t"]
out_rig = sim_rig["OUT"]
names_rig = sim_rig["Z_NAMES"]
U_rig = sim_rig["U"]

out_ode = sim_sur["OUT"]


idx_wo_rig = names_rig.index("w_o_out")
idx_wginj_rig = names_rig.index("w_G_inj")

# idx_wo_ode = names_ode.index("w_o_out")
# idx_wginj_ode = names_ode.index("w_G_inj")

# u2 is the second control
u2_rig = U_rig[:, 1]

# ============================================================
# Figure
# ============================================================

fig, axs = plt.subplots(
    3,
    1,
    figsize=(7.2, 7.0),   # good for full-page LaTeX figure
    sharex=True,
)

# ------------------------------------------------------------
# 1) Oil production rate
# ------------------------------------------------------------
#
axs[0].plot(
    t_rig,
    out_rig[:, idx_wo_rig],
    color="blue",
    linestyle="-",
    linewidth=1.6,
    label="High-fidelity Simulator (DAE)",
)

axs[0].plot(
    t_rig,
    out_ode[:, idx_wo_rig],
    color="red",
    linestyle="--",
    linewidth=1.4,
    label="Surrogate ODE",
)

axs[0].set_ylabel(r"$w_{o,\mathrm{out}}$ [kg/s]")
axs[0].grid(False)
axs[0].legend(frameon=False, loc="best")

# ------------------------------------------------------------
# 2) Injected gas flow rate
# ------------------------------------------------------------

axs[1].plot(
    t_rig,
    out_rig[:, idx_wginj_rig],
    color="blue",
    linestyle="-",
    linewidth=1.6,
    label="High-fidelity Simulator (DAE)",
)

axs[1].plot(
    t_rig,
    out_ode[:, idx_wginj_rig],
    color="red",
    linestyle="--",
    linewidth=1.4,
    label="Surrogate ODE",
)

axs[1].set_ylabel(r"$w_{G,\mathrm{inj}}$ [kg/s]")
axs[1].grid(False)

# ------------------------------------------------------------
# 3) Control input u2
# ------------------------------------------------------------
#
axs[2].plot(
    t_rig,
    u2_rig,
    color="gray",
    linestyle="-",
    linewidth=1.6,
)

axs[2].set_ylabel(r"$u_2$ [-]")
axs[2].set_xlabel(r"Time [s]")
axs[2].grid(False)

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

for ax in axs:
    ax.tick_params(direction="in")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

fig.align_ylabels(axs)

plt.tight_layout()

plt.savefig(
    "dynamic_comparison_full_page_2.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()

##############################################################
##############################################################


