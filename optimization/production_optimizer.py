# THIS PIECE OF SOFTWARE SOLVES THE OPTIMIZATION PROBLEM WITH N WELLS, COLLECTING THEIR STABILITY CONSTRAINT AUTOMATICALLY
# IT IS DESIGN TO OPTIMIZE WITH THE FOLLOWING MODELS:
    # RIGOROUS CASADI SIMULATOR MODELS
    # SURROGATE CASADI SIMULATOR MODELS
    # PINN CASADI SIMULATOR MODELS AS WELL
# IT DOES NOT USE A SURROGATE-BASED METHODOLOGY, IT OPTIMIZES WITH THE MODEL THAT IS PROVIDED
# IT IS DESIGNED AS A GENERIC ENGINE AND SHALL NOT BE CHANGED BY THE USER
# IT IS READY TO IMPLEMENT FOR AS MANY WELLS AS YOU WISH


###################################################################
################ GENERAL LIBRARIES USED ###########################
###################################################################

import casadi as ca
import numpy as np

###################################################################
################ OTHER PACKAGES NEEDED ###########################
###################################################################


from application.simulation_engine import make_model
import numpy as np
from configuration.wells import get_wells


def print_solution(sol, show_states=False, show_algebraic=False):
    Z_NAMES = sol["Z_NAMES"]

    print("\n" + "="*60)
    print("SOLVER STATUS")
    print("="*60)

    stats = sol["stats"]
    print("Return status:", stats.get("return_status", "N/A"))
    print("Success      :", stats.get("success", "N/A"))
    print("Iterations   :", stats.get("iter_count", "N/A"))

    print("\n" + "="*60)
    print("TOTALS")
    print("="*60)
    for k, v in sol["totals"].items():
        print(f"{k:20s}: {v:.6f}")

    print("\n" + "="*60)
    print("PER-WELL RESULTS")
    print("="*60)

    for w in sol["per_well"]:
        print("\n" + "-"*50)
        print(f"Well: {w['well_name']}")
        print("-"*50)

        # Controls
        u = np.array(w["u"]).reshape(-1)
        print("u* =", u)

        # States
        if show_states:
            y = np.array(w["y"]).reshape(-1)
            print("y* =", y)

        # Algebraic states
        if show_algebraic and w["z"] is not None:
            z = np.array(w["z"]).reshape(-1)
            print("z* =", z)

        # Outputs
        print("\nOutputs:")
        out_dict = w["out_dict"]

        for name in Z_NAMES:
            val = out_dict.get(name, None)
            if val is not None:
                print(f"{name:25s}: {val:.6f}")

        # Key indicators
        print("\nKey indicators:")
        for key in ["w_o_out", "P_bh_bar", "P_tb_b_bar"]:
            if key in out_dict:
                print(f"{key:25s}: {out_dict[key]:.6f}")


def print_compact_summary(sol):
    print("\n" + "="*60)
    print("COMPACT SUMMARY")
    print("="*60)

    for w in sol["per_well"]:
        u = np.array(w["u"]).reshape(-1)
        out = w["out_dict"]

        print(
            f"{w['well_name']:5s} | "
            f"u = {u} | "
            f"w_o = {out.get('w_o_out', np.nan):.4f} | "
            f"P_bh = {out.get('P_bh_bar', np.nan):.2f} | "
            f"P_tb = {out.get('P_tb_b_bar', np.nan):.2f}"
        )


def _build_out_dict(out_vec, Z_NAMES):
    return {name: out_vec[i] for i, name in enumerate(Z_NAMES)}

def polyval_casadi(u1, coef):
    val = 0
    deg = len(coef) - 1
    for k, c in enumerate(coef):
        val += float(c) * u1**(deg - k)
    return val


def optimize_field_production(
        model_type: str,
        wells: dict,
        # -------------------------
        # Initial guesses (lists of lists)
        # -------------------------
        u_guess_list=None,
        # -------------------------
        # Coupling capacities (optional)
        # -------------------------
        G_available=None,
        G_max_export=None,
        W_max=None,
        L_max=None,
        # -------------------------
        # Individual well constraints
        # -------------------------
        P_max_tb_b_bar=120,
        P_min_bh_bar=90,
        # -------------------------
        # User definition
        # -------------------------
        unconstrained_well=False,
        unconstrained_platform=False,
        enforce_stable=True,
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

    INF=1e20
    # ---------------------
    # 1) Define how many wells are there
    # ---------------------
    well_names=list(wells.keys())
    N=len(well_names)

    # ---------------------
    # 2) Build one well to know dimensions and names
    # ---------------------
    first_well_name = next(iter(wells))
    first_well_data = wells[first_well_name]

    model=make_model(
        model_type,
        BSW=first_well_data["BSW"],
        GOR=first_well_data["GOR"],
        PI=first_well_data["PI"],
        K_gs=first_well_data["K_gs"],
        K_inj=first_well_data["K_inj"],
        K_pr=first_well_data["K_pr"])
    is_dae=bool(model["is_dae"])
    nx=int(model["nx"])
    nu=int(model["nu"])
    nz = int(model["nz"]) if is_dae else 0
    Z_NAMES=model["Z_NAMES"]

    # ---------------------
    # 3) Build all models
    # ---------------------
    models = {}

    if is_dae:
        for well_name, well_data in wells.items():
            models[well_name] = make_model(
                model_type,
                BSW=well_data["BSW"],
                GOR=well_data["GOR"],
                PI=well_data["PI"],
                K_gs=well_data["K_gs"],
                K_inj=well_data["K_inj"],
                K_pr=well_data["K_pr"]
            )

    else:
        for well_name, well_data in wells.items():
            models[well_name] = make_model(
                model_type,
                BSW=well_data["BSW"],
                GOR=well_data["GOR"],
                PI=well_data["PI"],
                K_gs=well_data["K_gs_sur"],
                K_inj=well_data["K_inj_sur"],
                K_pr=well_data["K_pr_sur"]
            )

    # ---------------------
    # 4) Ipopt_opts
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
    # 5) Bounds and defaults
    # ---------------------

    if is_dae:
        y_lb=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        y_ub=[INF,INF,INF,INF,INF,INF,INF]
        z_lb = [1e5, 1e5, 0.0, 0.0] # 1 bar and 0 kg/s
        z_ub = [5e7, 5e7, 50, 50] # 500 bar and 50 ks/s
    else:
        y_lb=[0.0,0.0,0.0]
        y_ub=[INF,INF,INF]
    u_lb=[0.05,0.10]
    u_ub=[1.0,1.0]

    # ---------------------
    # 6) Decision Variables (symbolic, stacked)
    # ---------------------

    Y=[]
    U=[]
    Z=[]
    Xblocks=[]

    for j in range(N):
        y_j=ca.MX.sym(f"y_{j}",nx)
        u_j=ca.MX.sym(f"u_{j}",nu)
        Y.append(y_j)
        U.append(u_j)

        if is_dae:
            z_j=ca.MX.sym(f"z_{j}",nz)
            Z.append(z_j)
            Xblocks.append(ca.vertcat(y_j,z_j,u_j))
        else:
            Xblocks.append(ca.vertcat(y_j,u_j))

    x=ca.vertcat(*Xblocks)

    # ---------------------
    # 7) Constraints and objective
    # ---------------------

    g_list,lbg,ubg=[],[],[]

    # Aggregated (coupling) totals
    total_w_o=0 # total oil produced
    total_w_w=0 # total water produced
    total_w_g_inj=0 # total gas injected
    total_w_g_res=0 # total gas leaving reservoir
    total_w_l_out=0 # total liquid out

    # Build each well block and add constraints
    for j, well_name in enumerate(well_names):
        well_data=wells[well_name]
        well_model=models[well_name]

        BSW=well_data["BSW"]

        F_all=well_model["F_all"]

        if is_dae:
            dx_j,g_j,out_j=F_all(Y[j],Z[j],U[j])
        else:
            dx_j,out_j=F_all(Y[j],U[j])
            g_j=None
        outD=_build_out_dict(out_j,Z_NAMES)

        # totals
        total_w_o+=outD["w_o_out"] # total oil produced
        total_w_w+=outD["w_w_out"] # total water produced
        total_w_g_inj+=outD["w_G_inj"]  # total gas injected
        total_w_g_res+=outD["w_G_res"] # total gas leaving reservoir
        total_w_l_out+=outD["w_L_out"] # total liquid out

        # ---------------------
        # 7.1) Per well constraint
        # ---------------------

        # 7.1.1) Steady State

        g_list.append(dx_j)
        lbg+=[0.0]*nx
        ubg+=[0.0]*nx

        if is_dae:
            g_list.append(g_j)
            lbg+=[0.0]*nz
            ubg+=[0.0]*nz

        if not unconstrained_well:

            # 7.1.2) Minimum bottom-hole pressure

            g_list.append(outD["P_bh_bar"])
            lbg.append(float(P_min_bh_bar))
            ubg.append(INF)

            # 7.1.3) Maximum tubing pressure

            g_list.append(outD["P_tb_b_bar"])
            lbg.append(0)
            ubg.append(float(P_max_tb_b_bar))

        # 7.1.4) Per-well health/validity constraints

        # alpha constraints in [0,1]
        # This will avoid non-physical solutions and render the possible infeasible if no feasible exists
        for akey in ["alpha_L_tb_b", "alpha_L_tb_t", "alpha_L_tb"]:
            g_list.append(outD[akey])
            lbg.append(0.0)
            ubg.append(1.0)

        Vmin_g = 1e-12 # minimum gas cushion
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

        # forward pressure drops
        for dp_key in ["dP_tb_choke_bar", "dP_gs_an_bar", "dP_res_bh_bar", "dP_int_bar"]:
            g_list.append(outD[dp_key])
            lbg.append(0.0)
            ubg.append(INF)

        # nonnegative flows
        for wkey in ["w_res","w_up","w_G_inj", "w_out", "w_L_out", "w_G_out"]:
            g_list.append(outD[wkey])
            lbg.append(0.0)
            ubg.append(1e20)

        # 7.1.5) Per-well stability inequality
        if enforce_stable:
            coef_i = well_data["coeff_stability"]

            u1_j = U[j][0]
            u2_j = U[j][1]

            b_i = polyval_casadi(u1_j, coef_i)

            g_list.append(b_i - u2_j)
            lbg.append(-INF)
            ubg.append(0.0)

    # ---------------------
    # 7.2) Objective
    # ---------------------
    #
    obj=total_w_o

    # obj=0

    # ---------------------
    # 7.3) Coupling Constraints (field capacities)
    # ---------------------

    if not unconstrained_platform:
        if G_available is not None:
            g_list.append(total_w_g_inj)
            lbg.append(0.0)
            ubg.append(float(G_available))

        if G_max_export is not None:
            g_list.append(total_w_g_res)
            lbg.append(0.0)
            ubg.append(float(G_max_export))

        if W_max is not None:
            g_list.append(total_w_w)
            lbg.append(0.0)
            ubg.append(float(W_max))

        if L_max is not None:
            g_list.append(total_w_l_out)
            lbg.append(0.0)
            ubg.append(float(L_max))

    g = ca.vertcat(*g_list)

    # -------------------------
    # 7.4) Bounds and initial guess
    # -------------------------

    DEFAULT_U = [0.90, 0.90]
    lbx_list=[]
    ubx_list=[]
    x0_list=[]

    for j, well_name in enumerate(well_names):
        well_data=wells[well_name]

        # ---------------------
        # y (from wells)
        # ---------------------
        if is_dae:
            y_guess = list(well_data["y_guess_rig"])
        else:
            y_guess = list(well_data["y_guess_sur"])


        lbx_list+=list(y_lb)
        ubx_list+=list(y_ub)
        x0_list += y_guess

        # ---------------------
        # z (DAE only)
        # ---------------------
        if is_dae:

            z_guess = list(well_data["z_guess_rig"])

            lbx_list+=list(z_lb)
            ubx_list+=list(z_ub)
            x0_list += z_guess

        # ---------------------
        # u (controls)
        # ---------------------
        if u_guess_list is not None:
            u_guess=list(u_guess_list[j])
        else:
            u_guess=list(DEFAULT_U)

        lbx_list+=list(u_lb)
        ubx_list+=list(u_ub)
        x0_list+=u_guess

    # Convert to CasADi
    lbx = ca.DM(lbx_list).reshape((-1, 1))
    ubx = ca.DM(ubx_list).reshape((-1, 1))
    x0 = ca.DM(x0_list).reshape((-1, 1))

    # -------------------------
    # 8) Solve NLP
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
    # 9) Decode per-well solution + evaluate outputs
    # -------------------------

    per_well=[]
    idx=0

    tot_oil = 0.0
    tot_water = 0.0
    tot_gas_inj = 0.0
    tot_g_exp = 0.0
    tot_liquid_prod = 0.0

    for j, well_name in enumerate(well_names):
        yj=x_star[idx:idx+nx]
        idx+=nx
        zj=None
        if is_dae:
            zj=x_star[idx:idx+nz]
            idx+=nz
        uj=x_star[idx:idx+nu]
        idx+=nu

        well_model = models[well_name]
        F_all = well_model["F_all"]

        if is_dae:
            dx_j,g_j,out_j=F_all(yj,zj,uj)
        else:
            dx_j,out_j=F_all(yj,uj)
        out_j=ca.DM(out_j)
        out_num = {name: float(out_j[k]) for k, name in enumerate(Z_NAMES)}

        tot_oil += out_num.get("w_o_out", 0.0)
        tot_water += out_num.get("w_w_out", 0.0)
        tot_gas_inj += out_num.get("w_G_inj", 0.0)
        tot_g_exp += out_num.get("w_G_res", 0.0)
        tot_liquid_prod += out_num.get("w_L_out", 0.0)

        per_well.append({
            "well_name": well_name,
            "y":yj,
            "z":zj,
            "u":uj,
            "dx":ca.DM(dx_j),
            "g": ca.DM(g_j) if is_dae else None,
            "out":out_j,
            "out_dict":out_num,
        })
    totals= {
        "w_o_out": tot_oil,
        "w_w_out": tot_water,
        "w_G_inj": tot_gas_inj,
        "w_G_out": tot_g_exp,
        "w_L_out": tot_liquid_prod,
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

if __name__=="__main__":

    wells =get_wells()

    sol = optimize_field_production(
        model_type="rigorous",
        wells=wells,
        G_available=14.00,
        G_max_export=1.40,
        W_max=11.50,
        L_max=40,
        unconstrained_well=False,
        unconstrained_platform=False,
        enforce_stable=True
    )

    print_solution(sol, show_states=True, show_algebraic=True)
    print_compact_summary(sol)

    w_o_out = sol["per_well"][0]["out_dict"]["w_o_out"]
    P_bh    = sol["per_well"][0]["out_dict"]["P_bh_bar"]
    P_tb    = sol["per_well"][0]["out_dict"]["P_tb_b_bar"]