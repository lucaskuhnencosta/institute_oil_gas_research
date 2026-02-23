import casadi as ca

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
        "alpha_G_m_tb_b",
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


def build_steady_state_model(f_func,
                            state_size=3,
                            control_size=2,
                            alg_size=None,
                            name="glc_ss"):
    """
        Build symbolic CasADi blocks for steady-state models.

        ODE surrogate expected signature:
            dx, out = f_func(y,u)

        DAE rigorous expected signature (alg_size must be provided):
            dx,g,out=f_func(y,z_alg,u)

        Returns a dict. Check model ["is DAE"]

    """
    Z_names=Z_NAMES
    is_dae=alg_size is not None

    # ---------------------
    # 1) Symbols
    # ---------------------
    y = ca.MX.sym("y", state_size)
    u = ca.MX.sym("u", control_size)
    if is_dae:
        z_alg=ca.MX.sym("z_alg", alg_size)
        z_alg_names=[f"z_alg{i}" for i in range(alg_size)]
        # ---------------------
        # 2) Model evaluation
        # ---------------------
        dx,g,out=f_func(y, z_alg, u)
        # Residual for steady-state solve
        res=ca.vertcat(dx,g)

        # ---------------------
        # 3) Jacobians
        # ---------------------
        yz = ca.vertcat(y, z_alg)
        J_yz = ca.jacobian(res, yz)
        J_u = ca.jacobian(res, u)

        # --- Reduced Jacobian for stability: Ared = fx - fz * (gz \ gx)
        fx = ca.jacobian(dx, y)
        fz = ca.jacobian(dx, z_alg)
        gx = ca.jacobian(g,  y)
        gz = ca.jacobian(g,  z_alg)

        Ared=fx-fz@ca.solve(gz,gx)

        # ---------------------
        # 4) CasADi Functions
        # ---------------------
        F_all = ca.Function(
            f"{name}_all",
            [y, z_alg, u],
            [dx, g, out],
            ["y", "z_alg", "u"],
            ["dx", "g", "out"],
        )

        F_res = ca.Function(
            f"{name}_res",
            [y, z_alg, u],
            [res],
            ["y", "z_alg", "u"],
            ["res"],
        )

        F_J_yz = ca.Function(
            f"{name}_J_yz",
            [y, z_alg, u],
            [J_yz],
            ["y", "z_alg", "u"],
            ["J_yz"],
        )

        F_J_u = ca.Function(
            f"{name}_J_u",
            [y, z_alg, u],
            [J_u],
            ["y", "z_alg", "u"],
            ["J_u"],
        )

        F_A = ca.Function(
            f"{name}_Ared",
            [y, z_alg, u],
            [Ared],
            ["y", "z_alg", "u"],
            ["A"],
        )

        F_dx_out = ca.Function(
            f"{name}_dx_out",
            [y, z_alg, u],
            [dx, out],
            ["y", "z_alg", "u"],
            ["dx", "out"],
        )

        return {
            # Common
            "is_dae": True,
            "name": name,
            "y": y,
            "u": u,
            "dx": dx,
            "out": out,
            "F_all": F_all,
            "Z_names": Z_names,
            "nx": state_size,
            "nu": control_size,
            "F_A":F_A,

            #DAE-specific
            "z_alg": z_alg,
            "g": g,
            "res": res,
            "nz": alg_size,
            "Z_alg_names": z_alg_names,
            "F_res": F_res,
            "F_J_yz": F_J_yz,
            "F_J_u": F_J_u,
            "F_dx_out": F_dx_out,
        }
    else:
        # ---------------------
        # 2) Model evaluation
        # ---------------------
        dx,out=f_func(y,u)

        # Stability Jacobian
        A = ca.jacobian(dx, y)

        F_all = ca.Function(
            f"{name}_all",
            [y, u],
            [dx, out],
            ["y", "u"],
            ["dx", "out"])


        F_A = ca.Function(f"{name}_A",
                          [y, u],
                          [A],
                          ["y", "u"],
                          ["A"])

        return {
            # Common
            "is_dae": False,
            "name": name,
            "y": y,
            "u": u,
            "dx": dx,
            "out": out,
            "F_all": F_all,
            "Z_names": Z_names,
            "nx": state_size,
            "nu": control_size,
            "F_A": F_A,

            # ODE-specific
            "A": A,

        }

import numpy as np

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

"""

        You will solve for (y,z_alg) such that:
            dx(y,z_alg,u) = 0
            g (y,z_alg,u) = 0

        where:
            y      : differential states (nx)
            z_alg  : algebraic unknowns (nz_alg), e.g. [P_tb, P_bh, w_res]
            u      : controls/inputs (nu)

        f_dae_func must have signature:
            dx, g, out = f_dae_func(y, z_alg, u)
        returning:
            dx  : (nx x 1)
            g   : (nz_alg x 1)
            out : (nout x 1) any extra algebraic outputs/logging (can be empty)

        Returns dict with:
            y_sym, z_sym, u_sym
            dx_expr, g_expr, out_expr
            F_all(y,z,u) -> (dx,g,out)
            F_res(y,z,u) -> stacked residual [dx; g]
            J_yz(y,z,u)  -> Jacobian d([dx;g]) / d([y;z])
            (optionally) J_u(y,z,u) -> d([dx;g]) / d(u)

"""