import casadi as ca



def build_steady_state_model(f_func,
                            state_size,
                            control_size=2,
                            alg_size=None,
                            name="glc_ss",
                            out_name=None,
                            regularize_gz=True,
                            gz_reg=1e-8
                            ):
    """
        Build symbolic CasADi blocks for steady-state models.

        ODE surrogate expected signature:
            dx, out = f_func(y,u)

        DAE rigorous expected signature (alg_size must be provided):
            dx,g,out=f_func(y,z_alg,u)

        Returns a dict. Check model ["is DAE"]

    """
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

        ng = int(g.size1())  # number of algebraic equations
        nz = int(z_alg.size1())

        Ared = ca.MX.nan(fx.size1(), fx.size2())  # default placeholder

        if ng == nz:
            if regularize_gz:
                I = ca.MX.eye(nz)
                X = ca.solve(gz + gz_reg * I, gx)
            else:
                X = ca.solve(gz, gx)
            Ared = fx - fz @ X
        else:
        # Optional: still allow partial "Ared" only if gz is square
            pass


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
            "Z_names": out_name,
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
            "Ared": Ared,
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
            "Z_names": out_name,
            "nx": state_size,
            "nu": control_size,
            "F_A": F_A,

            # ODE-specific
            "A": A,

        }

# import numpy as np
#
# """
# Build symbolic CasADi functions for steady-state (equilibrium) solving of ODE or DAE.
# Actually builds symbolic blocks:
# - dx(y,u), z(y,u)
# - A(y,u) = d(dx)/d(y)
#
# f_func(y,u) must return either:
#   - dx
#   - or (dx, z)
#
# Returns a dict with:
#   y_sym, u_sym
#   dx_expr, z_expr
#   F_dx(y,u), F_all(y,u)
# """
#
# """
#
#         You will solve for (y,z_alg) such that:
#             dx(y,z_alg,u) = 0
#             g (y,z_alg,u) = 0
#
#         where:
#             y      : differential states (nx)
#             z_alg  : algebraic unknowns (nz_alg), e.g. [P_tb, P_bh, w_res]
#             u      : controls/inputs (nu)
#
#         f_dae_func must have signature:
#             dx, g, out = f_dae_func(y, z_alg, u)
#         returning:
#             dx  : (nx x 1)
#             g   : (nz_alg x 1)
#             out : (nout x 1) any extra algebraic outputs/logging (can be empty)
#
#         Returns dict with:
#             y_sym, z_sym, u_sym
#             dx_expr, g_expr, out_expr
#             F_all(y,z,u) -> (dx,g,out)
#             F_res(y,z,u) -> stacked residual [dx; g]
#             J_yz(y,z,u)  -> Jacobian d([dx;g]) / d([y;z])
#             (optionally) J_u(y,z,u) -> d([dx;g]) / d(u)
#
# """