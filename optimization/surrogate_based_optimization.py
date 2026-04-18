import numpy as np
np.set_printoptions(suppress=True, precision=6)
from utilities.block_builders import *
from simulators.black_box_simulator.experimental_truth_casadi import make_glc_well_rigorous, Z_NAMES_RIG
from optimization.api_interface_simulator import plant_zeroth_and_first_order

#####################################################################
# 0 - ALL FUNCTIONS DEFINITIONS
#####################################################################

def SBO(P_bh_min):

    def build_rk_derivatives(F_u2z):
        u = ca.MX.sym("u",2)
        y = F_u2z(u=u)["z"]   # r_k(u)
        J = ca.jacobian(y,u)  # ∂y/∂u  (3x2)
        F_rk = ca.Function(
            "F_rk",
            [u],
            [y,J],
            ["u"],
            ["z","J"]
        )
        return F_rk


    def solve_trust_region_subproblem(
            F_u2z,
            u_k,
            Delta,
            u_min,
            u_max,
            P_min_bh,
            P_max_tb_b,
            ipopt_opts=None,
    ):
        u_k=np.array(u_k,dtype=float).reshape((2,))


        # Decision variable (DOFs only)
        u=ca.MX.sym("u",2)

        z=F_u2z(u=u)["z"]
        m_o=z[0]
        P_bh=z[1]
        P_tb=z[2]

        g=[]
        lbg=[]
        ubg=[]

        # Inequality constraints
        g += [P_bh];      lbg += [P_min_bh];  ubg += [ca.inf]
        g += [P_tb];      lbg += [-ca.inf];   ubg += [P_max_tb_b]

        # Stability constraint u2 >= b_hat(u1)
        u1 = u[0]; u2 = u[1]
        b_hat = -0.3268*u1*u1 + 0.5116*u1 + 0.01914
        g += [u2 - b_hat]; lbg += [0.0]; ubg += [ca.inf]

        du=u-ca.DM(u_k)
        g+=[ca.dot(du,du)]
        lbg+=[-ca.inf]
        ubg+=[Delta**2]

        # Objective (IPOPT minimizes)
        f = -m_o

        nlp = {"x": u, "f": f, "g": ca.vertcat(*g)}
        if ipopt_opts is None:
            ipopt_opts = {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 2000,
                "ipopt.tol": 1e-10,
                "ipopt.constr_viol_tol": 1e-8,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps",
            }

        solver = ca.nlpsol("tr_solver", "ipopt", nlp, ipopt_opts)

        sol = solver(
            x0=ca.DM(u_k),
            lbx=ca.DM(u_min),
            ubx=ca.DM(u_max),
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
        )

        u_star = np.array(sol["x"]).reshape((2,))
        z_star = np.array(F_u2z(u=u_star)["z"]).reshape((-1,))

        return {
            "success": bool(solver.stats().get("success", False)),
            "stats": solver.stats(),
            "u_star": u_star,
            "m_o_out": float(z_star[0]),
            "P_bh": float(z_star[1]),
            "P_tb_b": float(z_star[2]),
        }

    def filter_accept(theta_trial, phi_trial, filter_list, gamma_theta=0.01, gamma_phi=0.01):
        """
        Return True if (theta_trial, phi_trial) is acceptable to the filter.
        filter_list is a list of dicts: [{"theta":..., "phi":...}, ...]
        """
        theta_trial = float(theta_trial)
        phi_trial = float(phi_trial)

        for e in filter_list:
            theta_i = float(e["theta"])
            phi_i   = float(e["phi"])

            cond_theta = theta_trial <= (1.0 - gamma_theta) * theta_i
            cond_phi   = phi_trial   <= phi_i - gamma_phi * theta_i

            # must satisfy at least one condition for each filter entry
            if not (cond_theta or cond_phi):
                return False

        return True

    def filter_update(theta_new, phi_new, filter_list, prune=False):
        """
        Add (theta_new, phi_new) to the filter.
        Optionally prune dominated entries.

        Dominance rule (minimize both):
          (theta_a, phi_a) dominates (theta_b, phi_b) if:
            theta_a <= theta_b and phi_a <= phi_b, with at least one strict.
        """
        theta_new = float(theta_new)
        phi_new = float(phi_new)

        filter_list.append({"theta": theta_new, "phi": phi_new})

        if not prune:
            return filter_list

        # prune dominated points
        kept = []
        for i, a in enumerate(filter_list):
            dominated = False
            for j, b in enumerate(filter_list):
                if i == j:
                    continue
                if (b["theta"] <= a["theta"] and b["phi"] <= a["phi"] and
                    (b["theta"] < a["theta"] or b["phi"] < a["phi"])):
                    dominated = True
                    break
            if not dominated:
                kept.append(a)

        # also remove near-duplicates
        uniq = []
        for e in kept:
            if not any(abs(e["theta"]-q["theta"]) < 1e-12 and abs(e["phi"]-q["phi"]) < 1e-12 for q in uniq):
                uniq.append(e)

        return uniq


    #####################################################################
    # 1 - SOLVER INITIALIZATION. RUNS JUST ONCE
    #####################################################################

    print("\n============= SOLVER INITIALIZATION START =============\n")

    #--------------------------------------------------------------------
    # 1.1 - OPTIMIZATION SETTINGS
    #--------------------------------------------------------------------

    u_min=[0.05,0.10]
    u_max=[1.00,1.00]
    umin = np.array(u_min, dtype=float).reshape((2,))
    umax = np.array(u_max, dtype=float).reshape((2,))
    P_min_bh=P_bh_min
    P_max_tb_b=120

    #--------------------------------------------------------------------
    # 1.2 - Z_NAMES AND REAL PLANT
    #--------------------------------------------------------------------

    Z_NAMES = Z_NAMES_RIG
    well_func = make_glc_well_rigorous(BSW=0.20, GOR=0.05, PI=3.0e-6)
    plant_model = build_steady_state_model(
        f_func=well_func,
        state_size=7,
        control_size=2,
        alg_size=4,
        name="glc_ss_rigorous",
        out_name=Z_NAMES
    )

    #--------------------------------------------------------------------
    # 1.3 - SURROGATE MODEL DEFINITIONS
    #--------------------------------------------------------------------

    pinn = PINN(
            hidden_units=[64, 64, 64])
    algnn = AlgNN(
        hidden_units=[64, 64, 64, 64])

    # 1) Load your torch models
    # from your_code import PINN, AlgNN, load_model_weights
    pinn = load_model_weights(pinn, "../training/PINN.pth")
    algnn = load_model_weights(algnn, "../training/AlgNN.pth")

    # 2) Extract weights
    pinn_w = extract_pinn_standard_weights(pinn)
    alg_w = extract_algnn_standard_weights(algnn)

    # 3) IMPORTANT: use the SAME scaling constants as the trained models
    # You can read them from buffers:
    pinn_y_min = pinn.y_min.cpu().numpy().tolist()
    pinn_y_max = pinn.y_max.cpu().numpy().tolist()
    pinn_u_min = pinn.u_min.cpu().numpy().tolist()
    pinn_u_max = pinn.u_max.cpu().numpy().tolist()

    alg_y_min = algnn.y_min.cpu().numpy().tolist()
    alg_y_max = algnn.y_max.cpu().numpy().tolist()
    alg_u_min = algnn.u_min.cpu().numpy().tolist()
    alg_u_max = algnn.u_max.cpu().numpy().tolist()
    alg_z_min = algnn.z_min.cpu().numpy().tolist()
    alg_z_max = algnn.z_max.cpu().numpy().tolist()

    F_pinn = build_casadi_pinn_function_standard(pinn_w,
                                                 pinn_y_min,
                                                 pinn_y_max,
                                                 pinn_u_min,
                                                 pinn_u_max)

    F_u2z = build_casadi_surrogate_u2z(
        pinn_weights=pinn_w,
        algnn_weights=alg_w,
        pinn_y_min=pinn_y_min, pinn_y_max=pinn_y_max,
        pinn_u_min=pinn_u_min, pinn_u_max=pinn_u_max,
        alg_y_min=alg_y_min, alg_y_max=alg_y_max,
        alg_u_min=alg_u_min, alg_u_max=alg_u_max,
        alg_z_min=alg_z_min, alg_z_max=alg_z_max,
    )
    F_rk = build_rk_derivatives(F_u2z)

    #--------------------------------------------------------------------
    # 1.4 - INITIAL GUESSES
    #--------------------------------------------------------------------

    y_guess_rig_init = [3679.08033973,
                        289.73390193,
                        3167.56224658,
                        1041.96126532,
                        50.46858403,
                        759.52720527,
                        249.84447542]

    z_guess_rig_init = [8.75897957e+06,
                        8.42155186e+06,
                        2.17230613e+01,
                        2.17230613e+01]

    u_k_init=[0.5,0.5]
    Delta_init=0.05

    print("\n============= SOLVER INITIALIZATION END =============")

    ###############################################################
    # 2 - BUILD THE MODEL R_K
    ###############################################################

    k=0
    filter_list=[]
    u_k=u_k_init
    y_guess_rig=y_guess_rig_init
    z_guess_rig=z_guess_rig_init
    Delta=Delta_init
    min_obj=np.inf
    max_infeasibility=-np.inf
    theta_tol=1e-6
    theta_k=None
    phi_plant_k=None

    max_iter=100

    history = {
        "k": [],
        "u_k": [],
        "u_trial": [],
        "Delta": [],
        "theta_k": [],
        "theta_trial": [],
        "phi_k": [],
        "phi_trial": [],
        "accepted": [],
        "step_type": [],  # "init", "f-type", "theta-type", "reject"
    }

    while k<max_iter:
        print("\n=====================================================")
        print(f"=============== ITERATION {k} ======================")
        print("=====================================================")
        print(f"We are at u_k={u_k}")
        u_k_arr = np.array(u_k, dtype=float).reshape((2,))

        res = F_rk(u=u_k)
        z_k = np.array(res["z"]).reshape((-1,))
        J_k = np.array(res["J"])

        print("\n----- SURROGATE LOCAL MODEL -----")
        print("r_k(u_k) = [w_o_out, P_bh_bar, P_tb_b_bar]:")
        print(z_k)

        print("\n∇r_k(u_k)  (Jacobian dy/du)")
        print(J_k)

        print("\nInterpretation:")
        print("Rows  = outputs [w_out,P_bh,P_tb_b]")
        print("Cols  = controls [u1,u2]")
        print("Entry (i,j) = ∂y_i / ∂u_j")
        print("-------------------------------------\n")

        simulation_import_api=plant_zeroth_and_first_order(
            model=plant_model,
            u_k=u_k,
            y_guess=y_guess_rig,
            z_guess=z_guess_rig
        )

        y_guess_rig=simulation_import_api["y_star"]
        z_guess_rig=simulation_import_api["z_star"]

        plant_z_k=simulation_import_api["z0"]
        phi_k_current = None if phi_plant_k is None else float(phi_plant_k)
        theta_k_current = None if theta_k is None else float(theta_k)

        plant_J_k=simulation_import_api["J"]

        print("\n----- TRUTH MODEL -----")

        print("Plant z0(u_k) = [w_o_out, P_bh_bar, P_tb_b_bar]:")
        print(plant_z_k)



        print("\nPlant J(u_k) = dz/du:")
        print(plant_J_k)

        print("\nInterpretation:")
        print("Rows  = outputs [w_out,P_bh,P_tb_b]")
        print("Cols  = controls [u1,u2]")
        print("Entry (i,j) = ∂y_i / ∂u_j")
        print("-------------------------------------\n")

        u = ca.MX.sym("u", 2)
        z_surr = F_u2z(u=u)["z"]  # (3x1)

        c_k = plant_z_k - z_k  # (3,)
        C_k = plant_J_k - J_k  # (3,2)

        du = u - ca.DM(u_k)
        z_corr = z_surr + ca.DM(c_k) + ca.DM(C_k) @ du

        F_u2z_corr = ca.Function(
            f"F_u2z_corr_{k}",
            [u],
            [z_corr],
            ["u"],
            ["z"]
        )

        print("\n--- CORRECTION TERMS at iteration ---")
        print("c_k      =", c_k)
        print("\nC_k =\n", C_k)
        print("------\n")

        if theta_k is not None and theta_k<theta_tol:
            print("\nAlgorithm converged")
            print(u_k)
            print(z_plant_trial)
            print(z_surr_trial)
            u_converged = np.array(u_k, dtype=float).reshape((2,))
            return {
                "u_converged": u_converged,
                "history": history,
            }
        print("------\n")

        print("\n--- Solving TRSP ---")

        ###############################################################
        # 3 - TRUST REGION SUBPROBLEM
        ###############################################################
        # F_u2z_corr is your corrected CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
        res = solve_trust_region_subproblem(
            F_u2z=F_u2z_corr,
            u_k=u_k,
            Delta=Delta,
            u_min=u_min,
            u_max=u_max,
            P_min_bh=P_min_bh,
            P_max_tb_b=P_max_tb_b
        )

        print("Solver success:", res["stats"]["success"])
        print("u* =", res["u_star"])
        print("m_o_out* =", res["m_o_out"])
        print("P_bh* =", res["P_bh"])
        print("P_tb_b* =", res["P_tb_b"])

        u_k=np.array(u_k,dtype=float).reshape((2,))
        z_k = np.array(F_u2z(u=u_k)["z"]).reshape((-1,))

        u_star=np.array(res["u_star"],dtype=float).reshape((2,))

        m_os, Pbh_s, Ptb_s = res["m_o_out"], res["P_bh"], res["P_tb_b"]
        print("m_os =", m_os)
        print("Pbh_s =", Pbh_s)
        print("Ptb_s =", Ptb_s)

        m_ok,Pbh_k,Ptb_k=z_k[0],z_k[1],z_k[2]

        #So what was the step size?
        step = u_star - u_k
        step_norm = float(np.linalg.norm(step))
        hit_TR = step_norm >= 0.999 * float(Delta)

        # Constraint slacks (>=0 means satisfied, =0 active)
        slack_Pbh = float(Pbh_s - P_min_bh)  # >= 0
        slack_Ptb = float(P_max_tb_b - Ptb_s)  # >= 0

        u1, u2 = float(u_star[0]), float(u_star[1])
        b_hat = -0.3268 * u1 * u1 + 0.5116 * u1 + 0.01914
        slack_stab = float(u2 - b_hat)  # >= 0

        # Improvement
        dm = float(m_os - m_ok)

        print("\n--- TRUST-REGION SUBPROBLEM STATISTICS ---")
        print("success:", res["success"], res["stats"].get("return_status", ""))
        print(f"u_k    = {u_k}")
        print(f"u_star = {u_star}")
        print(f"step   = {step} | ||step||={step_norm:.6f}  (Delta={Delta})  hit_TR={hit_TR}")

        print("\nObjective:")
        print(f"m_o_out(u_k)    = {m_ok:.6f}")
        print(f"m_o_out(u_star) = {m_os:.6f}")
        print(f"Δm_o_out        = {dm:.6f}")

        print("\nConstraints at u_star (slack >= 0 means satisfied):")
        print(f"P_bh     = {Pbh_s:.6f}  (>= {P_min_bh})  slack = {slack_Pbh:.6f}")
        print(f"P_tb_b   = {Ptb_s:.6f}  (<= {P_max_tb_b})  slack = {slack_Ptb:.6f}")
        print(f"stability u2-b_hat = {slack_stab:.6f}  (>= 0)")
        print("------------------------------------------------------------\n")

        ###############################################################
        # 4 - FILTER
        ###############################################################
        print("\n--- FILTER CHECK ---")
        u_trial=u_star

        z_surr_trial = np.array(F_u2z_corr(u=u_trial)["z"]).reshape((-1,))
        print(f"Z_surr_trial =",z_surr_trial)

        plant_res_trial=plant_zeroth_and_first_order(
            model=plant_model,
            u_k=u_trial,
            y_guess=y_guess_rig,
            z_guess=z_guess_rig)

        y_guess_rig = plant_res_trial["y_star"]
        z_guess_rig = plant_res_trial["z_star"]

        z_plant_trial=plant_res_trial["z0"]
        print(f"z_plant_trial =",z_plant_trial)

        phi_plant_trial=-float(z_plant_trial[0])

        print(f"The objective we are are minimizing is {phi_plant_trial:.6f} at trial whereas our minimum is {min_obj}")
        if phi_plant_trial<min_obj:
           min_obj=phi_plant_trial

        theta_vec = np.abs(z_plant_trial - z_surr_trial)  # (3,)
        theta_trial = float(np.linalg.norm(theta_vec, ord=2))  # Euclidean

        print("u_trial =", u_trial)
        print("theta_trial =", theta_trial)
        print("phi_trial (plant) =", phi_plant_trial)

        if k==0:
            accepted=True

            history["k"].append(k)
            history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
            history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
            history["Delta"].append(float(Delta))
            history["theta_k"].append(theta_k_current)
            history["theta_trial"].append(float(theta_trial))
            history["phi_k"].append(phi_k_current)
            history["phi_trial"].append(float(phi_plant_trial))
            history["accepted"].append(True)
            history["step_type"].append("init")

            u_k=u_trial.tolist()
            theta_k=theta_trial
            phi_plant_k = phi_plant_trial
            k = k + 1
            filter_list = filter_update(theta_k, phi_plant_k, filter_list, prune=False)
            continue
        else:
            accepted = filter_accept(
                theta_trial=theta_trial,
                phi_trial=phi_plant_trial,  # usually use PLANT values in the filter
                filter_list=filter_list
            )
        print("Accepted by filter?", accepted)
        if accepted:
            print(f"        --- Switching Condition ---")
            if phi_plant_k-phi_plant_trial>=0.01*(theta_k**0.9):
                print(f"        This is an f-type step")
                print(f"        That means that filter is not updated and trust-region increases")
                history["k"].append(k)
                history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
                history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
                history["Delta"].append(float(Delta))
                history["theta_k"].append(float(theta_k))
                history["theta_trial"].append(float(theta_trial))
                history["phi_k"].append(float(phi_plant_k))
                history["phi_trial"].append(float(phi_plant_trial))
                history["accepted"].append(True)
                history["step_type"].append("f-type")

                Delta=Delta*2
                u_k = u_trial.tolist()
                theta_k = theta_trial
                phi_plant_k = phi_plant_trial
                k = k + 1
                continue
            else:
                print(f"        This is an theta-type step")
                print(f"        That means that filter is updated and trust-region changes according to the ratio test")
                history["k"].append(k)
                history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
                history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
                history["Delta"].append(float(Delta))
                history["theta_k"].append(float(theta_k))
                history["theta_trial"].append(float(theta_trial))
                history["phi_k"].append(float(phi_plant_k))
                history["phi_trial"].append(float(phi_plant_trial))
                history["accepted"].append(True)
                history["step_type"].append("theta-type")
                u_k = u_trial.tolist()
                rho_k=1-(theta_trial/theta_k)
                if rho_k<0.4:
                    Delta=Delta*0.5
                elif rho_k<0.8:
                    Delta=Delta
                else:
                    Delta=Delta*2.0

                filter_list = filter_update(theta_k, phi_plant_k, filter_list, prune=False)
                phi_plant_k = phi_plant_trial
                theta_k = theta_trial
                k = k + 1

        else:
            # reject (typical action: shrink trust region Delta)
            print("Step failed")
            history["k"].append(k)
            history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
            history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
            history["Delta"].append(float(Delta))
            history["theta_k"].append(theta_k_current)
            history["theta_trial"].append(float(theta_trial))
            history["phi_k"].append(phi_k_current)
            history["phi_trial"].append(float(phi_plant_trial))
            history["accepted"].append(False)
            history["step_type"].append("reject")

            Delta *= 0.5
            k=k+1
    return {
        "u_converged": None,
        "history": history,
    }