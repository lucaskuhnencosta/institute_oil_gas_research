import casadi as ca
import numpy as np

def build_steady_state_model(f_func, state_size=3, control_size=2, name="glc_ss"):
    """
    Build symbolic CasADi functions for steady-state (equilibrium) solving.

    f_func(y,u) must return either:
      - dx
      - or (dx, z)

    Returns a dict with:
      y_sym, u_sym
      dx_expr, z_expr
      F_dx(y,u), F_all(y,u)
    """
    y = ca.MX.sym("y", state_size)
    u = ca.MX.sym("u", control_size)

    out = f_func(y, u)
    dx, z = out

    F_dx  = ca.Function(f"{name}_dx",  [y, u], [dx], ["y", "u"], ["dx"])
    F_all = ca.Function(f"{name}_all", [y, u], [dx, z], ["y", "u"], ["dx", "z"])

    return {
        "y": y,
        "u": u,
        "dx": dx,
        "z": z,
        "F_dx": F_dx,
        "F_all": F_all,
        "nx":state_size,
        "nu": control_size,
        "name":name,
    }


# def build_ipopt_equilibrium_solver(
#     f_func,
#     state_size=3,
#     control_size=2,
#     y_lb=None,
#     y_ub=None,
#     eps_vgas=1e-6,
#     dx_scale=None,
#     reg_y=1e-8,
#     name="glc_eq"
# ):
#     """
#     Builds an IPOPT NLP solver to find steady-state y for a given u by minimizing ||dx||^2.
#
#     Decision variable: y
#     Parameter: p = [u; y_guess]  (so you can warm-start & regularize toward guess)
#     Constraints: optional physical constraints (vgas, rhomix) if z contains them, or
#                  you can compute them inside f_func and include in z.
#
#     Returns:
#       solver (nlpsol), and helper function unpack_p(p)->(u, y_guess)
#     """
#     # symbolic variables
#     y = ca.MX.sym("y", state_size)
#     u = ca.MX.sym("u", control_size)
#     y_guess = ca.MX.sym("y_guess", state_size)
#
#     out = f_func(y, u)
#     if isinstance(out, (tuple, list)) and len(out) == 2:
#         dx, z = out
#     else:
#         dx = out
#         z = ca.MX.zeros(0, 1)
#
#     # scaling for residual
#     if dx_scale is None:
#         dx_scale = np.ones(state_size)
#     dx_scale = ca.DM(dx_scale).reshape((state_size, 1))
#     dx_scaled = dx / dx_scale
#
#     # objective: weighted least squares + small regularization
#     J = ca.dot(dx_scaled, dx_scaled) + reg_y * ca.dot(y - y_guess, y - y_guess)
#
#     # constraints vector
#     g = []
#     lbg = []
#     ubg = []
#
#     Vgas = z[2]
#     g.append(Vgas)
#     lbg.append(eps_vgas)
#     ubg.append(1e20)
#
#     g_expr = ca.vertcat(*g)
#
#     # bounds on y
#     if y_lb is None: y_lb = [0]*state_size
#     if y_ub is None: y_ub = [1e20]*state_size
#
#     # pack parameters: p = [u; y_guess]
#     p = ca.vertcat(u, y_guess)
#
#     nlp = {"x": y, "p": p, "f": J, "g": g_expr}
#
#     opts = {
#         "ipopt.print_level": 0,
#         "print_time": 0,
#         "ipopt.max_iter": 500,
#         # helps with nasty scaling problems
#         "ipopt.nlp_scaling_method": "gradient-based",
#         "ipopt.mu_strategy": "adaptive",
#     }
#
#     solver = ca.nlpsol(name, "ipopt", nlp, opts)
#
#     def pack_p(u_val, y_guess_val):
#         return ca.vertcat(ca.DM(u_val).reshape((control_size,1)),
#                           ca.DM(y_guess_val).reshape((state_size,1)))
#
#     return solver, pack_p, y_lb, y_ub, lbg, ubg
