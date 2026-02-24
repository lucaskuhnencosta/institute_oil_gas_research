import importlib

from Surrogate_ODE_Model.glc_surrogate_casadi import *
from Rigorous_DAE_model.glc_rigorous_casadi import glc_well_01_rigorous_casadi

from Utilities.block_builders import build_steady_state_model
import casadi as ca
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


def _build_out_dict(out_vec, Z_NAMES):
    return {name: out_vec[i] for i, name in enumerate(Z_NAMES)}

def _get_module_builder(module_name: str):
    """
    module_name must expose:
        - build_well_model(i) -> dict with keys:
            is_dae, nx, nu, nz, Z_NAMES, F_all
        - Z_NAMES (optional, but usually present)
    """
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "build_well_model"):
        raise AttributeError(f"Module '{module_name}' must define build_well_model(i).")
    return mod.build_well_model

def optimize_field_production(
        module_name: str,
        N: int,
        # -------------------------
        # Initial guesses (lists of lists)
        # -------------------------
        y_guess_list,
        u_guess_list,
        z_guess_list=None,
        # -------------------------
        # Coupling capacities (optional)
        # -------------------------
        G_available=None,
        G_max_export_burn=None,
        W_max=None,
        L_max=None,
        O_max=None,
        # -------------------------
        # Individual well constraints
        # -------------------------
        P_max_tb_b_bar=120,
        P_min_bh_bar=85
        ):

    """
       Solves:
         maximize oil -> minimize -w_o_out

       Decision variables:
         ODE: x = [y; u]
         DAE: x = [y; z; u]

       Returns:
         y_star, u_star, (z_star if DAE), dx_star, (g_star if DAE), out_star,
         eig, stable, stats
       """

    # ---------------------
    # 1) Build one well to know dimensions and names
    # ---------------------

    build_well_model = _get_module_builder(module_name)
    proto=build_well_model(i=0)
    is_dae=bool(proto["is_dae"])
    nx=int(proto["nx"])
    nu=int(proto["nu"])
    nz = int(proto["nz"]) if is_dae else 0
    Z_NAMES=proto["Z_NAMES"]

    # ---------------------
    # 2) User can change ipopt opts here
    # ---------------------

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
    # 3) Bounds and defaults
    # ---------------------

    y_lb=[0.0]*nx
    y_ub=[1e20]*nx
    u_lb=[0.0]*nu
    u_ub=[1.0]*nu
    if is_dae:
        z_lb = [1e3, 1e3, 0.0]
        z_ub = [1e9, 1e9, 1e6]

    # ---------------------
    # 4) Decision Variables (symbolic, stacked)
    # ---------------------

    Y=[]
    U=[]
    Z=[]
    Xblocks=[]

    for i in range(N):
        y_i=ca.MX.sym(f"y_{i}",nx)
        u_i=ca.MX.sym(f"u_{i}",nu)
        Y.append(y_i)
        U.append(u_i)

        if is_dae:
            z_i=ca.MX.sym(f"z_{i}",nz)
            Z.append(z_i)
            Xblocks.append(ca.vertcat(y_i,z_i,u_i))
        else:
            Xblocks.append(ca.vertcat(y_i,u_i))

    x=ca.vertcat(*Xblocks)

    # ---------------------
    # 5) Constraints and objective
    # ---------------------

    g_list,lbg,ubg=[],[],[]

    # Aggregated (coupling) totals
    total_w_o=0
    total_w_w=0
    total_w_g_inj=0
    total_w_g_out=0
    total_w_l_out=0

    # Build each well block and add constraints
    for i in range(N):
        well=build_well_model(i=i)
        F_all=well["F_all"]
        if is_dae:
            dx_i,g_i,out_i=F_all(Y[i],Z[i],U[i])
        else:
            dx_i,out_i=F_all(Y[i],U[i])
            g_i=None
        outD=_build_out_dict(out_i,Z_NAMES)

        # totals
        total_w_o+=outD["w_o_out"]
        total_w_w+=outD["w_w_out"]
        total_w_g_inj+=outD["w_G_inj"]
        total_w_g_out+=outD["w_G_out"]
        total_w_l_out+=outD["w_L_out"]

        # ---------------------
        # 5.1) Per well constraint
        # ---------------------

        # 5.1.1) Steady State

        g_list.append(dx_i)
        lbg+=[0.0]*nx
        ubg+=[0.0]*nx

        if is_dae:
            g_list.append(g_i)
            lbg+=[0.0]*nz
            ubg+=[0.0]*nz

        # 5.1.2) Minimum bottom-hole pressure

        g_list.append(outD["P_bh_bar"])
        lbg.append(float(P_min_bh_bar))
        ubg.append(1e20)

        # 5.1.3) Maximum tubing pressure

        g_list.append(outD["P_tb_b_bar"])
        lbg.append(0)
        ubg.append(float(P_max_tb_b_bar))

        # 5.1.4) Per-well health/validity constraints

        # log args must stay positive
        g_list.append(outD["log_arg_tb"])
        lbg.append(1e-12)
        ubg.append(1e20)

        g_list.append(outD["log_arg_bh"])
        lbg.append(1e-12)
        ubg.append(1e20)

        # alpha constraints in [0,1]
        for akey in ["alpha_L_tb_b", "alpha_L_tb_t", "alpha_G_tb_t", "alpha_avg_L_tb"]:
            g_list.append(outD[akey])
            lbg.append(0.0)
            ubg.append(1.0)

        # pressures positive
        for pkey in ["P_an_t_bar", "P_an_b_bar", "P_tb_t_bar", "P_tb_b_bar", "P_bh_bar"]:
            g_list.append(outD[pkey])
            lbg.append(1e-6)
            ubg.append(1e20)

        # choke forward
        g_list.append(outD["dP_tb_choke_bar"])
        lbg.append(0.0)
        ubg.append(1e20)

        # gas source -> annulus
        g_list.append(outD["dP_gs_an_bar"])
        lbg.append(0.0)
        ubg.append(1e20)

        # reservoir -> well
        g_list.append(outD["dP_res_bh_bar"])
        lbg.append(0.0)
        ubg.append(1e20)

        # nonnegative flows
        for wkey in ["w_G_inj", "w_out", "w_L_out", "w_G_out"]:
            g_list.append(outD[wkey])
            lbg.append(0.0)
            ubg.append(1e20)

        # 5.1.5) Per-well stability inequality

        if i==0:
            u1=U[i][0]
            u2=U[i][1]
            b_hat=-0.4969*u1*u1+0.7561*u1+0.195
            g_stab=u2-b_hat
            g_list.append(g_stab)
            lbg.append(0.0)
            ubg.append(1e20)
        #... need to implement for all wells, this will be later

    # ---------------------
    # 5.2) Objective
    # ---------------------

    obj=-total_w_o

    # ---------------------
    # 5.3) Coupling Constraints (field capacities)
    # ---------------------

    # if G_inj_max is not None:
    #     g_list.append(total_w_g_inj)
    #     lbg.append(0.0)
    #     ubg.append(float(G_inj_max))
    #
    # if G_sep_max is not None:
    #     g_list.append(total_w_g_out)
    #     lbg.append(0.0)
    #     ubg.append(float(G_sep_max))
    #
    # if W_max is not None:
    #     g_list.append(total_w_w)
    #     lbg.append(0.0)
    #     ubg.append(float(W_max))
    #
    # if L_max is not None:
    #     g_list.append(total_w_l_out)
    #     lbg.append(0.0)
    #     ubg.append(float(L_max))
    #
    # if O_max is not None:
    #     g_list.append(total_w_o)
    #     lbg.append(0.0)
    #     ubg.append(float(O_max))

    g = ca.vertcat(*g_list)

    # -------------------------
    # 5.4) Bounds and initial guess
    # -------------------------

    lbx_list=[]
    ubx_list=[]
    x0_list=[]

    for i in range(N):
        # y
        lbx_list+=list(y_lb)
        ubx_list+=list(y_ub)
        x0_list += list(y_guess_list[i])

        # z
        if is_dae:
            if z_guess_list is None:
                raise ValueError("DAE requires z_guess_list (length N).")
            lbx_list+=list(z_lb)
            ubx_list+=list(z_ub)
            x0_list += list(z_guess_list[i])

        # u
        lbx_list+=list(u_lb)
        ubx_list+=list(u_ub)
        x0_list+=list(u_guess_list[i])

    lbx = ca.DM(lbx_list).reshape((-1, 1))
    ubx = ca.DM(ubx_list).reshape((-1, 1))
    x0 = ca.DM(x0_list).reshape((-1, 1))

    # -------------------------
    # 6) Solve NLP
    # -------------------------
    nlp = {"x": x, "f": obj, "g": g}
    solver=ca.nlpsol("field_solver", "ipopt", nlp,ipopt_opts)

    sol=solver(
        x0=x0,
        lbx=lbx,
        ubx=ubx,
        lbg=ca.DM(lbg),
        ubg=ca.DM(ubg)
    )
    stats=solver.stats()
    x_star=sol["x"]

    # -------------------------
    # 7) Decode per-well solution + evaluate outputs
    # -------------------------

    per_well=[]
    idx=0

    tot_o = 0.0
    tot_w = 0.0
    tot_gi = 0.0
    tot_go = 0.0
    tot_lo = 0.0

    for i in range(N):
        yi=x_star[idx:idx+nx]
        idx+=nx
        zi=None
        if is_dae:
            zi=x_star[idx:idx+nz]
            idx+=nz
        ui=x_star[idx:idx+nu]
        idx+=nu

        well = build_well_model(i)
        F_all = well["F_all"]

        if is_dae:
            dx_i,g_i,out_i=F_all(yi,zi,ui)
        else:
            dx_i,out_i=F_all(yi,ui)
        out_i=ca.DM(out_i)
        out_num = {name: float(out_i[k]) for k, name in enumerate(Z_NAMES)}

        tot_o += out_num.get("w_o_out", 0.0)
        tot_w += out_num.get("w_w_out", 0.0)
        tot_gi += out_num.get("w_G_inj", 0.0)
        tot_go += out_num.get("w_G_out", 0.0)
        tot_lo += out_num.get("w_L_out", 0.0)

        per_well.append({
            "i":i+1,
            "y":yi,
            "z":zi,
            "u":ui,
            "dx":ca.DM(dx_i),
            "g": ca.DM(g_i) if is_dae else None,
            "out":out_i,
            "out_dict":out_num,
        })
    totals= {
        "w_o_out": tot_o,
        "w_w_out": tot_w,
        "w_G_inj": tot_gi,
        "w_G_out": tot_go,
        "w_L_out": tot_lo,
    }
    return {
        "stats": stats,
        "is_dae": is_dae,
        "nx": nx,
        "nu": nu,
        "nz": nz,
        "Z_NAMES": Z_NAMES,
        "totals": totals,
        "per_well": per_well,
        "x_star": x_star,
    }

res = optimize_field_production(
    module_name="Surrogate_ODE_Model.glc_surrogate_casadi",
    N=1,
    y_guess_list=[[3919.7688, 437.16663, 7956.1206]],
    u_guess_list=[[0.20, 0.55]],
    z_guess_list=None,  # ODE => None
    P_max_tb_b_bar=130,
    P_min_bh_bar=90,
)

print("success:", res["stats"]["success"], res["stats"]["return_status"])
print("totals:", res["totals"])
print("u* well1:", np.array(res["per_well"][0]["u"]).squeeze())
print("y* well1:", np.array(res["per_well"][0]["y"]).squeeze())


res = optimize_field_production(
    module_name="Rigorous_DAE_model.glc_rigorous_casadi",
    N=1,
    y_guess_list=[[3919.7688, 437.16663, 7956.1206]],
    u_guess_list=[[0.20, 0.55]],
    z_guess_list=[[120e5, 140e5, 10.0]],  # [P_tb_b, P_bh, w_res] initial guess
    P_max_tb_b_bar=130,
    P_min_bh_bar=90,
)

print("success:", res["stats"]["success"], res["stats"]["return_status"])
print("totals:", res["totals"])
print("u* well1:", np.array(res["per_well"][0]["u"]).squeeze())
print("y* well1:", np.array(res["per_well"][0]["y"]).squeeze())
print("z* well1:", np.array(res["per_well"][0]["z"]).squeeze())

#
#
# # 1) assemble model
# model = build_steady_state_model(glc_casadi,
#                                  state_size=3,
#                                  control_size=2,
#                                  name="glc")
#
# # 2) solve one point
# u = [0.20, 0.05]
# # y_guess=[3582.4731,311.7586,8523.038]
# y_guess = [3919.7688, 437.16663, 7956.1206]
#
# y_star, u_star,dx_star, z_star, eig,stable,stats = optimize_production(
#     model=model,
#     u_val=u,
#     y_guess=y_guess,
# )
#
# print("status:", stats["return_status"], "success:", stats["success"])
# print("y*:", np.array(y_star).squeeze())
# print("u*:", np.array(u_star).squeeze())
# print("dx*:", np.array(dx_star).squeeze())
# print("||dx||:", np.linalg.norm(np.array(dx_star).squeeze()))
# # print("z*:", np.array(z_star).squeeze())
# print("eig:", eig)
# print("stable:", stable)
# print("\n--- z* (named) ---")
# Z_NAMES=model["Z_names"]
# print_z_grouped(z_star, Z_NAMES)  # set ncols=1 if you prefer
#
# #
# # y_sym = ca.MX.sym("y", nx)
# # u_sym = ca.MX.sym("u", nu)
# # if is_dae:
# #     z_sym = ca.MX.sym("z", nz)
# #     dx_sym, g_sym, out_sym = well_func(y_sym, z_sym, u_sym)
# #     F_all = ca.Function("F_all", [y_sym, z_sym, u_sym], [dx_sym, g_sym, out_sym])
#
# # def well_builder(i):
# #      """
# #      Return a well model for well i.
# #      Here you can parameterize wells by using different function closures,
# #      or different parameter sets per well.
# #
# #      IMPORTANT: all must share same Z_NAMES layout.
# #      """
# #      if MODEL_TYPE == "ode":
# #          return build_steady_state_model(
# #              model_type="ode",
# #              f_ode=glc_well_01_casadi,
# #              nx=3, nu=2,
# #              name=f"well{i}",
# #              Z_NAMES=Z_NAMES
# #          )
# #      else:
# #          return build_steady_state_model(
# #              model_type="dae",
# #              f_dae=glc_well_01_rigorous_casadi,
# #              nx=3, nu=2, nz=3,
# #              name=f"well{i}",
# #              Z_NAMES=Z_NAMES
# #          )
