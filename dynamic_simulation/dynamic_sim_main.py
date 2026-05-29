import numpy as np
import casadi as ca

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
        y_lb=None,
        y_ub=None,
        z_lb=None,
        z_ub=None,
        enforce_volume_constraints=True,
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

    if not model["is_dae"]:
        raise ValueError("This simulator is for DAE models only.")

    nx = model["nx"]
    nz = model["nz"]
    nu = model["nu"]

    # Decision variables
    y_next = ca.MX.sym("y_next", nx)
    z_next = ca.MX.sym("z_next", nz)
    x = ca.vertcat(y_next, z_next)

    # Parameters
    y_prev = ca.MX.sym("y_prev", nx)
    u_next = ca.MX.sym("u_next", nu)
    p = ca.vertcat(y_prev, u_next)

    # Model evaluation at next time
    dx_next, g_next, out_next = model["F_all"](y_next, z_next, u_next)

    # Backward Euler residual
    r_dyn = y_next - y_prev - dt * dx_next
    r_alg = g_next

    g_list = [r_dyn, r_alg]
    lbg = [0.0] * nx + [0.0] * nz
    ubg = [0.0] * nx + [0.0] * nz

    # Optional physical volume constraints
    if enforce_volume_constraints:
        # These are the same volume constants used in your steady-state solver.
        Vmin_g = 1e-12
        V_tb = 24.99334640648408
        V_bh = 2.356194490192345
        rho_o = 760.0
        rho_w = 1000.0

        # y = [
        #   m_G_an, m_G_t, m_o_t, m_w_t,
        #   m_G_b,  m_o_b, m_w_b
        # ]
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
    if y_lb is None:
        y_lb = [0.0] * nx
    if y_ub is None:
        y_ub = [1e20] * nx

    if z_lb is None:
        z_lb = [1e5, 1e5, 0.0, 0.0]
    if z_ub is None:
        z_ub = [5e7, 5e7, 50.0, 50.0]

    lbx = ca.DM(y_lb + z_lb)
    ubx = ca.DM(y_ub + z_ub)

    return solver, lbx, ubx, ca.DM(lbg), ca.DM(ubg)


def simulate_dae_backward_euler(
        model,
        y0,
        z0,
        u_fun,
        t_grid,
        Z_NAMES=None,
        y_lb=None,
        y_ub=None,
        z_lb=None,
        z_ub=None,
        enforce_volume_constraints=True,
        verbose=True,
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

    if not model["is_dae"]:
        raise ValueError("simulate_dae_backward_euler expects a DAE model.")

    nx = model["nx"]
    nz = model["nz"]
    nu = model["nu"]

    t_grid = np.array(t_grid, dtype=float).reshape(-1)
    nt = len(t_grid)

    if nt < 2:
        raise ValueError("t_grid must contain at least two time points.")

    y0 = np.array(y0, dtype=float).reshape(nx)
    z0 = np.array(z0, dtype=float).reshape(nz)

    # Storage
    Y = np.zeros((nt, nx))
    Z = np.zeros((nt, nz))
    U = np.zeros((nt, nu))

    DX = np.zeros((nt, nx))
    G = np.zeros((nt, nz))

    OUT = []
    OUT_DICT = []

    Y[0, :] = y0
    Z[0, :] = z0
    U[0, :] = np.array(u_fun(t_grid[0]), dtype=float).reshape(nu)

    # Evaluate initial point
    dx0, g0, out0 = model["F_all"](
        ca.DM(Y[0, :]),
        ca.DM(Z[0, :]),
        ca.DM(U[0, :])
    )

    DX[0, :] = np.array(dx0, dtype=float).reshape(nx)
    G[0, :] = np.array(g0, dtype=float).reshape(nz)

    out0_np = np.array(out0, dtype=float).reshape(-1)
    OUT.append(out0_np)

    if Z_NAMES is not None:
        OUT_DICT.append(make_output_dict(out0, Z_NAMES))

    # Time stepping
    for k in range(nt - 1):
        t_prev = t_grid[k]
        t_next = t_grid[k + 1]
        dt = float(t_next - t_prev)

        if dt <= 0:
            raise ValueError("t_grid must be strictly increasing.")

        u_next = np.array(u_fun(t_next), dtype=float).reshape(nu)

        solver, lbx, ubx, lbg, ubg = build_dae_backward_euler_step_solver(
            model=model,
            dt=dt,
            y_lb=y_lb,
            y_ub=y_ub,
            z_lb=z_lb,
            z_ub=z_ub,
            enforce_volume_constraints=enforce_volume_constraints,
            solver_name=f"dae_be_step_{k}"
        )

        # Warm start with previous solution
        x0 = ca.DM(np.concatenate([Y[k, :], Z[k, :]]))
        p = ca.DM(np.concatenate([Y[k, :], u_next]))

        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p,
        )

        stats = solver.stats()

        # if not bool(stats.get("success", False)):
        #     status = stats.get("return_status", "UNKNOWN")
        #     raise RuntimeError(
        #         f"DAE step failed at k={k}, "
        #         f"t={t_prev:.6g} -> {t_next:.6g}. "
        #         f"IPOPT status: {status}"
        #     )

        x_next = np.array(sol["x"], dtype=float).reshape(-1)

        y_next = x_next[:nx]
        z_next = x_next[nx:nx + nz]

        Y[k + 1, :] = y_next
        Z[k + 1, :] = z_next
        U[k + 1, :] = u_next

        dx_next, g_next, out_next = model["F_all"](
            ca.DM(y_next),
            ca.DM(z_next),
            ca.DM(u_next)
        )

        DX[k + 1, :] = np.array(dx_next, dtype=float).reshape(nx)
        G[k + 1, :] = np.array(g_next, dtype=float).reshape(nz)

        out_next_np = np.array(out_next, dtype=float).reshape(-1)
        OUT.append(out_next_np)

        if Z_NAMES is not None:
            OUT_DICT.append(make_output_dict(out_next, Z_NAMES))

        if verbose and ((k + 1) % max(1, (nt - 1) // 10) == 0):
            print(f"Completed step {k + 1}/{nt - 1}, t = {t_next:.3f}")

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

# Initial control
u0 = np.array([0.50, 0.50])

well="P1"
params=wells[well]
BSW = params["BSW"]
GOR = params["GOR"]
PI = params["PI"]
K_gs = params["K_gs"]
K_inj = params["K_inj"]
K_pr = params["K_pr"]
y_guess_rig = params["y_guess_rig"]
z_guess_rig = params["z_guess_rig"]

model_rig = make_model("rigorous",
                   BSW=BSW,
                   GOR=GOR,
                   PI=PI,
                   K_gs=K_gs,
                   K_inj=K_inj,
                   K_pr=K_pr)

# Find consistent initial condition using your robust steady-state solver
# y_ss, z_ss, dx_ss, g_ss, out_ss, eig, stable, stats = solve_equilibrium_ipopt(
#     model=model_rig,
#     u_val=u0,
#     y_guess=y_guess_rig,
#     z_guess=z_guess_rig,
# )

y_ss=y_guess_rig
z_ss=z_guess_rig

# Step change in production choke opening u1
import numpy as np

def u_fun(t):
    if t < 2000.0:
        return np.array([0.4, 0.4])
    elif t < 4000.0:
        return np.array([0.8, 0.8])
    elif t < 6000.0:
        return np.array([0.8, 0.6])


dt = 0.5
t_grid = np.arange(0.0, 8000 + dt, dt)

sim = simulate_dae_backward_euler(
    model=model_rig,
    y0=y_ss,
    z0=z_ss,
    u_fun=u_fun,
    t_grid=t_grid,
    Z_NAMES=Z_NAMES,
    enforce_volume_constraints=True,
    verbose=True,
)

import matplotlib.pyplot as plt

t = sim["t"]
out = sim["OUT"]
names = sim["Z_NAMES"]

idx_pbh = names.index("P_bh_bar")
idx_wout = names.index("w_o_out")
idx_wginj = names.index("w_G_inj")

plt.figure()
plt.plot(t, out[:, idx_pbh])
plt.xlabel("Time")
plt.ylabel("P_bh [bar]")
plt.grid(True)

plt.figure()
plt.plot(t, out[:, idx_wout], label="w_out")
plt.plot(t, out[:, idx_wginj], label="w_G_inj")
plt.xlabel("Time")
plt.ylabel("Flow [kg/s]")
plt.legend()
plt.grid(True)

plt.show()