from Rigorous_ODE_Model.glc_casadi import glc_casadi
from Utilities.block_builders import build_steady_state_model
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def solve_equilibrium_ipopt(
        model, # Ourput of a CasADi model (steady-state) that has been assembled
        u_val, #list/array shape (nu,)
        y_guess, #list/array shape (nx,)
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

    # ---------------------
    # 3) Parameter (fixed control)
    # ---------------------
    u_par=ca.MX.sym("u_par",nu)

    # ---------------------
    # 4) Create a function from original expressions, then re-call it
    # ---------------------
    F_all=ca.Function("F_all_internal",[y_sym,u_sym],[dx_expr,z_expr])
    dx_var,z_var=F_all(y_var,u_par) #Symbolic dx,z as functions of y_var, u_par

    # ---------------------
    # 5) Objective: minimize ||dx||^2 (scaled)
    # ---------------------
    obj=ca.dot(dx_var,dx_var)

    p=u_par

    # ---------------------
    # 6) Constraints
    # ---------------------

    g_list=[]
    lbg=[]
    ubg=[]

    V_gas=z_var[11]
    g_list.append(V_gas)
    lbg.append(1e-3)
    ubg.append(1e20)

    g=ca.vertcat(*g_list)

    # ---------------------
    # 7) Bounds on y
    # ---------------------
    y_lb=[0.0]*nx
    y_ub=[1e20]*nx

    lbx=ca.DM(y_lb).reshape((nx,1))
    ubx=ca.DM(y_ub).reshape((nx,1))

    # ---------------------
    # 8) NLP definition
    # ---------------------
    nlp={"x":y_var,"p":p,"f":obj,"g":g}

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    }

    solver = ca.nlpsol("eq_solver", "ipopt", nlp, opts)

    # -------------------------
    # Pack numeric inputs
    # -------------------------
    u_val = ca.DM(u_val).reshape((nu, 1))
    y0 = ca.DM(y_guess).reshape((nx, 1))
    p_val=u_val
    # -------------------------
    # Solve
    # -------------------------
    sol = solver(
        x0=y0,
        lbx=lbx,
        ubx=ubx,
        lbg=ca.DM(lbg) if len(lbg) else ca.DM([]),
        ubg=ca.DM(ubg) if len(ubg) else ca.DM([]),
        p=p_val,
    )

    y_star = sol["x"]

    # -------------------------
    # Evaluate dx,z at solution (numerically)
    # -------------------------
    # Reuse F_all we created (it maps y,u -> dx,z)
    dx_star, z_star = F_all(y_star, u_val)

    return y_star, dx_star, z_star, solver.stats()



# 1) assemble model
model = build_steady_state_model(glc_casadi, state_size=3, control_size=2, name="glc")

# 2) solve one point
u = [0.9, 0.2]
# y_guess=[3582.4731,311.7586,8523.038]
y_guess = [3919.7688, 437.16663, 7956.1206]

y_star, dx_star, z_star, stats = solve_equilibrium_ipopt(
    model=model,
    u_val=u,
    y_guess=y_guess,
)

print("status:", stats["return_status"], "success:", stats["success"])
print("y*:", np.array(y_star).squeeze())
print("dx*:", np.array(dx_star).squeeze())
print("||dx||:", np.linalg.norm(np.array(dx_star).squeeze()))
print("z*:", np.array(z_star).squeeze())
