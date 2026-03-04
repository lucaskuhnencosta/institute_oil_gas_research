import importlib

import casadi as ca
import numpy as np
from collections import defaultdict
from Solvers.solve_glc_ode_equilibrium import print_z_grouped
from Rigorous_DAE_model.glc_truth_casadi import Z_NAMES_RIG
from Surrogate_ODE_Model.glc_surrogate_casadi import Z_NAMES_SUR


from Application.model_analysis_application import make_model


def _build_out_dict(out_vec, Z_NAMES):
    return {name: out_vec[i] for i, name in enumerate(Z_NAMES)}

def optimize_field_production(
        model_type: str,
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
        P_min_bh_bar=90
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

    model=make_model(model_type)
    is_dae=bool(model["is_dae"])
    nx=int(model["nx"])
    nu=int(model["nu"])
    nz = int(model["nz"]) if is_dae else 0
    Z_NAMES=model["Z_NAMES"]

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
    if is_dae:
        y_lb=[0.0,0.0,0.0,0,0,0,0]
        y_ub=[1e20,1e20,1e20,1e20,1e20,1e20,1e20]
        z_lb = [1e5, 1e5, 0.0, 0.0]
        z_ub = [5e7, 5e7, 50, 50]
    else:
        y_lb=[0.0,0.0,0.0]
        y_ub=[1e20,1e20,1e20]
    u_lb=[0.0,0.0]
    u_ub=[1.0,1.0]



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
        well=model
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

        # alpha constraints in [0,1]
        for akey in ["alpha_L_tb_b", "alpha_L_tb_t", "alpha_L_tb"]:
            g_list.append(outD[akey])
            lbg.append(0.0)
            ubg.append(1.0)

        BSW=0.20
        Vmin_g = 1e-12  # gas cushion
        rho_o = 760.0
        rho_w = 1000.0
        rho_L=rho_o*(1-BSW)+BSW*rho_w
        D_bh = 0.2
        L_bh = 75.0
        S_bh = ca.pi * D_bh ** 2 / 4.0
        V_bh = S_bh * L_bh

        D_tb = 0.134
        L_tb = 1973.0
        S_tb = ca.pi * D_tb ** 2 / 4.0
        V_tb = S_tb * L_tb
        if is_dae:
            V_L_t = outD["m_o_t"] / rho_o + outD["m_w_t"] / rho_w
            V_L_b = outD["m_o_b"] / rho_o + outD["m_w_b"] / rho_w
        else:
            V_L_t = outD["m_o_t"] / rho_L

        # enforce: V_L <= V - Vmin_g
        g_list.append(V_L_t)
        lbg.append(0.0)
        ubg.append(float(V_tb - Vmin_g))

        if is_dae:
            g_list.append(V_L_b)
            lbg.append(0.0)
            ubg.append(float(V_bh - Vmin_g))

        # pressures positive
        # for pkey in ["P_an_t_bar", "P_an_b_bar", "P_tb_t_bar", "P_tb_b_bar", "P_bh_bar"]:
        #     g_list.append(outD[pkey])
        #     lbg.append(1e-6)
        #     ubg.append(1e20)

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

        # reservoir -> well
        g_list.append(outD["dP_int_bar"])
        lbg.append(0.0)
        ubg.append(1e20)

        # nonnegative flows
        for wkey in ["w_res","w_up","w_G_inj", "w_out", "w_L_out", "w_G_out"]:
            g_list.append(outD[wkey])
            lbg.append(0.0)
            ubg.append(1e20)

        # 5.1.5) Per-well stability inequality

        if i==0:
            u1=U[i][0]
            u2=U[i][1]
            b_hat=-0.3268*u1*u1+0.5116*u1+0.01914
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

        well = model
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
    model_type="surrogate",
    N=1,
    y_guess_list=[[3285.42, 300.822, 6910.91]],
    u_guess_list=[[1.00, 1.00]],
    z_guess_list=None,
    P_max_tb_b_bar=130,
    P_min_bh_bar=90,
)

print("success:", res["stats"]["success"], res["stats"]["return_status"])
print("totals:", res["totals"])
print("u* well1:", np.array(res["per_well"][0]["u"]).squeeze())
print("y* well1:", np.array(res["per_well"][0]["y"]).squeeze())
print_z_grouped(res["per_well"][0]["out"], Z_NAMES_SUR)

res = optimize_field_production(
    model_type="rigorous",
    N=1,
    y_guess_list=[[3679.08033973,
           289.73390193,
           3167.56224658,
           1041.96126532,
           50.46858403,
           759.52720527,
           249.84447542]],
    u_guess_list=[[1.00, 1.00]],
    z_guess_list=[[8.75897957e+06,
           8.42155186e+06,
           2.17230613e+01,
           2.17230613e+01]],  # [P_tb_b, P_bh, w_res] initial guess
    P_max_tb_b_bar=130,
    P_min_bh_bar=90,
)

print("success:", res["stats"]["success"], res["stats"]["return_status"])
print("totals:", res["totals"])
print("u* well1:", np.array(res["per_well"][0]["u"]).squeeze())
print("y* well1:", np.array(res["per_well"][0]["y"]).squeeze())
print("z* well1:", np.array(res["per_well"][0]["z"]).squeeze())
print_z_grouped(res["per_well"][0]["out"], Z_NAMES_RIG)
