import casadi as ca
import numpy as np

def build_steady_state_model(f_func, state_size=3, control_size=2, name="glc_ss"):
    """
    Build symbolic CasADi functions for steady-state (equilibrium) solving of ODE or DAE.
    Actually builds symbolic blocks:
    - dx(y,u), z(y,u)
    - A(y,u) = d(dx)/d(y)

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

    A=ca.jacobian(dx,y)

    F_all = ca.Function(f"{name}_all", [y, u], [dx, z], ["y", "u"], ["dx", "z"])
    F_A=ca.Function(f"{name}_A", [y, u], [A], ["y", "u"],["A"])

    Z_NAMES = [
        # --- Eq 1-4
        "P_an_t_bar",
        "P_an_b_bar",
        "rho_G_an_b",
        "rho_G_in",

        # --- Eq 5
        "dP_gs_an_bar",
        "w_G_in_original",
        "w_G_in",

        # --- Eq 6-7
        "V_gas_tb_t",
        "V_gas_tb_t_safe",
        "rho_G_tb_t",
        "P_tb_t_bar",

        # --- Eq 8-13
        "rho_avg_mix_tb",
        "alpha_avg_L_tb",
        "alpha_G_tb_b",
        "U_avg_L_tb",
        "denom_G",
        "denom_G_safe",
        "U_avg_G_tb",
        "U_avg_mix_tb",

        # --- Eq 14-16
        "Re_tb",
        "Re_tb_safe",
        "log_arg_tb",
        "log_arg_tb_safe",
        "lambda_tb",
        "F_t_bar",

        # --- Eq 17-18
        "P_tb_b_bar",
        "dP_an_tb_bar",
        "w_G_inj",

        # --- Eq 19-23
        "U_avg_L_bh",
        "Re_bh",
        "log_arg_bh",
        "lambda_bh",
        "F_bh_bar",
        "P_bh_bar",

        # --- Eq 24-27
        "dP_res_bh_bar",
        "w_res",
        "w_L_res",
        "w_G_res",
        "rho_G_tb_b",

        # --- Eq 28-30
        "denom_alpha_b",
        "denom_alpha_b_safe",
        "alpha_L_tb_b",
        "alpha_L_tb_t",
        "rho_mix_tb_t",
        "rho_mix_tb_t_safe",

        # --- Eq 31-35
        "dP_tb_choke_bar",
        "w_out",
        "Q_out",
        "denom_alpha_t",
        "denom_alpha_t_safe",
        "alpha_G_tb_t",
        "w_G_out",
        "w_L_out",

        "w_w_out",
        "w_o_out"
    ]

    return {
        "y": y,
        "u": u,
        "dx": dx,
        "z": z,
        "F_A": F_A,
        "F_all": F_all,
        "nx":state_size,
        "nu": control_size,
        "name":name,
        "Z_names": Z_NAMES
    }
