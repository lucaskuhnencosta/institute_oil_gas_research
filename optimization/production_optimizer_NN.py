import casadi as ca
from utilities.block_builders import build_casadi_surrogate_u2z_for_well
from optimization.production_optimizer import print_solution, print_compact_summary, polyval_casadi
from configuration.wells import get_wells

def optimize_field_production_nn(
        wells: dict,
        # -------------------------
        # Initial guesses
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
        P_max_tb_b_bar=120.0,
        P_min_bh_bar=90.0,
        # -------------------------
        # User definition
        # -------------------------
        unconstrained_well=False,
        unconstrained_platform=False,
        enforce_stable=True,
):
    """
    Field optimization using the composed NN surrogate u -> z.

    Per-well surrogate output ordering:
        z = [
            P_bh_bar,
            P_tb_b_bar,
            w_G_inj,
            w_res,
            w_L_res,
            w_G_res,
            w_w_out,
            w_o_out
        ]

    Decision variables:
        x = stacked controls only = [u_0; u_1; ...; u_{N-1}]

    Objective:
        maximize total oil production
    """

    INF = 1e20

    # ---------------------
    # 1) Define wells
    # ---------------------
    well_names = list(wells.keys())
    N = len(well_names)
    nu = 2

    # Final surrogate output names
    Z_NAMES = [
        "P_bh_bar",
        "P_tb_b_bar",
        "w_G_inj",
        "w_res",
        "w_L_res",
        "w_G_res",
        "w_w_out",
        "w_o_out",
    ]
    nz = len(Z_NAMES)

    # ---------------------
    # 2) Build per-well NN surrogates
    # ---------------------
    models = {}
    for well_name in well_names:
        models[well_name] = build_casadi_surrogate_u2z_for_well(well_name)

    # ---------------------
    # 3) IPOPT options
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
    # 4) Bounds on controls
    # ---------------------
    u_lb = [0., 0.0]
    u_ub = [1.0, 1.0]

    # ---------------------
    # 5) Decision variables
    # ---------------------
    U = []
    Xblocks = []

    for j in range(N):
        u_j = ca.MX.sym(f"u_{j}", nu)
        U.append(u_j)
        Xblocks.append(u_j)

    x = ca.vertcat(*Xblocks)

    # ---------------------
    # 6) Constraints and objective
    # ---------------------
    g_list, lbg, ubg = [], [], []

    total_w_o = 0
    total_w_w = 0
    total_w_g_inj = 0
    total_w_g_res = 0
    total_w_l_out = 0

    for j, well_name in enumerate(well_names):
        well_data = wells[well_name]
        F_u2z = models[well_name]

        z_j = F_u2z(u=U[j])["z"]

        names = [
            "P_bh_bar",   # 0
            "P_tb_b_bar", # 1
            "w_G_inj",    # 2
            "w_res",      # 3
            "w_L_res",    # 4
            "w_G_res",    # 5
            "w_w_out",    # 6
            "w_o_out"     # 7
        ]

        outD = {
            "P_bh_bar": z_j[0],
            "P_tb_b_bar": z_j[1],
            "w_G_inj": z_j[2],
            "w_res": z_j[3],
            "w_L_res": z_j[4],
            "w_G_res": z_j[5],
            "w_w_out": z_j[6],
            "w_o_out": z_j[7],
            "w_L_out": z_j[6] + z_j[7],
            # "w_G_out": z_j[2] + z_j[5],  # gas inj + gas from reservoir
        }

        # totals
        total_w_o += outD["w_o_out"]
        total_w_w += outD["w_w_out"]
        total_w_g_inj += outD["w_G_inj"]
        total_w_g_res += outD["w_G_res"]
        total_w_l_out += outD["w_L_out"]

        # ---------------------
        # 6.1) Per-well constraints
        # ---------------------
        if not unconstrained_well:
            # minimum bottom-hole pressure
            g_list.append(outD["P_bh_bar"])
            lbg.append(float(P_min_bh_bar))
            ubg.append(INF)

            # maximum tubing-bottom pressure
            g_list.append(outD["P_tb_b_bar"])
            lbg.append(0.0)
            ubg.append(float(P_max_tb_b_bar))

        # nonnegative learned/reconstructed flows
        for wkey in ["w_res", "w_G_inj", "w_G_res", "w_w_out", "w_o_out", "w_L_out"]:
            g_list.append(outD[wkey])
            lbg.append(0.0)
            ubg.append(INF)

        # stability inequality
        if enforce_stable:
            coef_i = well_data["coeff_stability"]
            u1_j = U[j][0]
            u2_j = U[j][1]
            b_i = polyval_casadi(u1_j, coef_i)

            g_list.append(b_i - u2_j)
            lbg.append(-INF)
            ubg.append(0.0)

    # ---------------------
    # 6.2) Objective
    # ---------------------
    obj = -total_w_o

    # ---------------------
    # 6.3) Coupling constraints
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

    g = ca.vertcat(*g_list) if g_list else ca.MX()

    # -------------------------
    # 7) Bounds and initial guess
    # -------------------------
    DEFAULT_U = [0.90, 0.90]
    lbx_list = []
    ubx_list = []
    x0_list = []

    for j, well_name in enumerate(well_names):
        if u_guess_list is not None:
            u_guess = list(u_guess_list[j])
        else:
            u_guess = list(DEFAULT_U)

        lbx_list += list(u_lb)
        ubx_list += list(u_ub)
        x0_list += u_guess

    lbx = ca.DM(lbx_list).reshape((-1, 1))
    ubx = ca.DM(ubx_list).reshape((-1, 1))
    x0 = ca.DM(x0_list).reshape((-1, 1))

    # -------------------------
    # 8) Solve NLP
    # -------------------------
    nlp = {"x": x, "f": obj, "g": g}
    solver = ca.nlpsol("field_solver_nn", "ipopt", nlp, ipopt_opts)

    sol = solver(
        x0=x0,
        lbx=lbx,
        ubx=ubx,
        lbg=ca.DM(lbg) if lbg else ca.DM(),
        ubg=ca.DM(ubg) if ubg else ca.DM(),
    )

    stats = solver.stats()
    x_star = sol["x"]

    # -------------------------
    # 9) Decode per-well solution
    # -------------------------
    per_well = []
    idx = 0

    tot_oil = 0.0
    tot_water = 0.0
    tot_gas_inj = 0.0
    tot_g_exp = 0.0
    tot_liquid_prod = 0.0

    for j, well_name in enumerate(well_names):
        uj = x_star[idx:idx + nu]
        idx += nu

        F_u2z = models[well_name]
        z_j = ca.DM(F_u2z(u=uj)["z"])

        out_num = {name: float(z_j[k]) for k, name in enumerate(Z_NAMES)}
        out_num["w_L_out"] = out_num["w_w_out"] + out_num["w_o_out"]
        out_num["w_G_out"] = out_num["w_G_inj"] + out_num["w_G_res"]

        tot_oil += out_num.get("w_o_out", 0.0)
        tot_water += out_num.get("w_w_out", 0.0)
        tot_gas_inj += out_num.get("w_G_inj", 0.0)
        tot_g_exp += out_num.get("w_G_res", 0.0)
        tot_liquid_prod += out_num.get("w_L_out", 0.0)

        per_well.append({
            "well_name": well_name,
            "y": None,
            "z": z_j,
            "u": uj,
            "dx": None,
            "g": None,
            "out": z_j,
            "out_dict": out_num,
        })

    totals = {
        "w_o_out": tot_oil,
        "w_w_out": tot_water,
        "w_G_inj": tot_gas_inj,
        "w_G_out": tot_g_exp,
        "w_L_out": tot_liquid_prod,
    }

    return {
        "stats": stats,
        "is_dae": False,
        "nx": 0,
        "nu": nu,
        "nz": nz,
        "Z_NAMES": Z_NAMES,
        "totals": totals,
        "per_well": per_well,
        "x_star": x_star,
    }

wells = get_wells()

sol = optimize_field_production_nn(
    wells=wells,
u_guess_list = [
    [0.17325437, 0.23448885],   # P1
    [0.58183577, 0.65922337]],
    G_available=14.00,
    G_max_export=1.40,
    W_max=11.50,
    L_max=40.0,
    unconstrained_well=False,
    unconstrained_platform=False,
    enforce_stable=True,
)

print_solution(sol, show_states=False, show_algebraic=True)
print_compact_summary(sol)