import numpy as np

from optimization.production_optimizer import polyval_casadi

np.set_printoptions(suppress=True, precision=6)
from utilities.block_builders import *

from optimization.api_interface_simulator import plant_zeroth_and_first_order


from simulators.black_box_simulator.black_box_model import make_glc_well_rigorous

#####################################################################
# 0 - ALL FUNCTIONS DEFINITIONS
#####################################################################

class SurrogateBasedOptimization:
    def __init__(self,
                 config,
                 wells,
                 rigorous_models,
                 F_u2z_models):
        """
        config: dict containing all problem settings
        """

        self.rigorous_models=rigorous_models
        self.F_u2z_models=F_u2z_models

        self.max_iter=config["max_iter"]


        self.nu=2

        self.wells=wells
        self.well_names = list(wells.keys())
        self.N = len(self.well_names)
        self.INF=10e20

        # -------------------------------
        # Problem settings
        # -------------------------------
        self.u_min = np.array(config["u_min"], dtype=float)
        self.u_max = np.array(config["u_max"], dtype=float)

        # -------------------------------
        # Well constrains
        # -------------------------------
        self.P_min_bh=float(config["P_min_bh"])
        self.P_max_tb_b=float(config["P_max_tb_b"])

        # -------------------------------
        # Global constraints
        # -------------------------------
        self.G_available=float(config["G_available"])
        self.G_max_export=float(config["G_max_export"])
        self.W_max=float(config["W_max"])
        self.L_max=float(config["L_max"])

        # -------------------------------
        # Constraint options
        # -------------------------------
        self.unconstrained_well=config["unconstrained_well"]
        self.unconstrained_platform=config["unconstrained_well"]
        self.enforce_stable=config["enforce_stable"]

        # -------------------------------
        # Surrogate-based optimization parameters
        # -------------------------------
        self.Delta=config["Delta"]
        self.gamma_c=config["gamma_c"] # Decrease of the trust region
        self.gamma_e=config["gamma_e"] # Increase of the trust region
        self.gamma_f=config["gamma_f"]
        self.gamma_theta=config["gamma_theta"]
        self.k_theta=config["k_theta"]
        self.gamma_s=config["gamma_s"]
        self.eta_1=config["eta_1"]
        self.eta_2=config["eta_2"]
        self.filter_list=[]

        self.theta_tol=config["theta_tol"]

        # self.F_u2z=config["F_u2z"]
        self.plant_eval=plant_zeroth_and_first_order ####### NEED TO CHECK THIS ONE HERE

        self.u0= config["u_guess_list"].reshape(-1)
        self.u_k=np.array(self.u0, dtype=float)

        self.theta_k=None
        self.phi_k=None

        self.u_lb=config["u_lb"]
        self.u_ub=config["u_ub"]

        self.u_guess_list=config["u_guess_list"]

        self.history=[]

        self.out_names=config["out_names"]

        self.ipopt_opts={
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 6000,
        "ipopt.tol": 1e-10,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def solve(self):
        k=0
        while k<self.max_iter:
            print("\n=====================================================")
            print(f"=============== ITERATION {k} ======================")
            print("=====================================================")
            print("u_k =", self.u_k)
            print("type(u_k) =", type(self.u_k))
            print("u_k shape =", np.shape(self.u_k))
            print("expected shape =", (self.N * self.nu,))

            # -------------------------------
            # 1. Evaluate surrogate + plant at current accepted point
            # -------------------------------
            z_surr_k_list, J_surr_k_list = self._eval_surrogates_with_jac(self.u_k) #This is ready
            z_plant_k_list, J_plant_k_list = self._eval_plant_models(self.u_k) # we need to build this API from scratch!

            print("\n----- SURROGATE LOCAL MODEL -----")
            print("r_k(u_k) = [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]:")
            print(z_surr_k_list)

            print("\n∇r_k(u_k)  (Jacobian dy/du)")
            print(J_surr_k_list)

            print("\nInterpretation:")
            print("Rows  = outputs [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]")
            print("Cols  = controls [u1,u2]")
            print("Entry (i,j) = ∂y_i / ∂u_j")
            print("-------------------------------------\n")

            print("\n----- PLANT VALUES -----")
            print("S(u_k) = [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]:")
            print(z_plant_k_list)

            print("\n∇r_k(u_k)  (Jacobian dy/du)")
            print(J_plant_k_list)

            print("\nInterpretation:")
            print("Rows  = outputs [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]")
            print("Cols  = controls [u1,u2]")
            print("Entry (i,j) = ∂y_i / ∂u_j")
            print("-------------------------------------\n")

            # -------------------------------
            # 2. Initialize the filter
            # -------------------------------
            if k==0 and self.theta_k is None and self.phi_k is None:
                phi_k=self._compute_phi(z_plant_k_list)
                theta_k,theta_k_per_well=self._compute_theta(z_plant_k_list,z_surr_k_list)

                self.phi_k=phi_k
                self.theta_k=theta_k
                self._filter_update(theta_k,phi_k)

                self.history.append({
                    "u": self.u_k.copy(),
                    "u_trial": self.u_k.copy(),
                    "Delta": self.Delta,
                    "theta": theta_k,
                    "theta_k_per_well": theta_k_per_well,
                    "phi": phi_k,
                    "type": "init"
                })

                print("\n--- FILTER INITIALIZATION ---")
                print("u0      =", self.u_k)
                print("phi_0   =", phi_k)
                print("theta_0 =", theta_k)
                print("filter  =", self.filter_list)

            # -------------------------------
            # 3. Convergence test at current accepted point
            # -------------------------------
            if self.theta_k <= self.theta_tol:
                oil_total = -self.phi_k
                print("\n=== CONVERGENCE ACHIEVED ===")
                print("theta_k=", self.theta_k)
                print("theta_tol=",self.theta_tol)
                print("u_converged=",self.u_k)
                print("oil_total =", oil_total)

                return {
                    "success": True,
                    "status": "converged",
                    "u_opt": self.u_k.copy(),
                    "theta_opt": self.theta_k,
                    "phi_opt": self.phi_k,
                    "oil_total": oil_total,
                    "iterations": k,
                    "history": self.history
                }
            # -------------------------------
            # 4. Build corrected models
            # -------------------------------
            F_corr_models = self._build_corrected_models( #This is done
                self.u_k,
                z_surr_k_list,
                J_surr_k_list,
                z_plant_k_list,
                J_plant_k_list
            )

            # -------------------------------
            # 5. Solve TRSP
            # -------------------------------
            trsp=self._solve_tr_subproblem(F_corr_models)
            u_trial = trsp["x_star"]

            # -------------------------------
            # STEP DIAGNOSTICS
            # -------------------------------
            du = u_trial - self.u_k
            step_norm = np.linalg.norm(du)

            print("\n--- TR STEP ---")
            print("u_k      =", self.u_k)
            print("u_trial  =", u_trial)
            print("du       =", du)
            print("||du||   =", step_norm)

            print("\n--- TRUST REGION ---")
            print("Delta          =", self.Delta)
            print("||du||^2       =", trsp["tr_norm_sq"])
            print("Delta^2        =", trsp["tr_radius_sq"])
            print("TR active?     =", trsp["tr_active"])

            # -------------------------------
            # 6. Evaluate trial point
            # -------------------------------

            z_surr_trial_list = self._eval_corrected_models(F_corr_models,u_trial)
            z_plant_trial_list, _ = self._eval_plant_models(u_trial)

            phi_trial = self._compute_phi(z_plant_trial_list)
            theta_trial,theta_trial_per_well = self._compute_theta(z_plant_trial_list, z_surr_trial_list)

            print("\n--- EVALUATING ACCEPTANCE ---")
            print("theta_trial =", theta_trial)
            print("theta_k=", self.theta_k)
            print("dtheta=",theta_trial-self.theta_k)
            print("phi_k       =", self.phi_k)
            print("phi_trial   =", phi_trial)
            print("dphi        =", phi_trial - self.phi_k)

            # -------------------------------
            # 7. Filter decision
            # -------------------------------
            accepted = self._filter_accept(theta_trial, phi_trial)

            if accepted:
                if self.phi_k - phi_trial >= self.k_theta * (self.theta_k ** self.gamma_s):
                    step_type = "f-type"
                    self.Delta *= self.gamma_e
                else:
                    step_type = "theta-type"

                    rho = 1 - theta_trial / self.theta_k

                    if rho < self.eta_1:
                        self.Delta *= self.gamma_c
                    elif rho > self.eta_2:
                        self.Delta *= self.gamma_e

                    self._filter_update(self.theta_k, self.phi_k)

                self._accept_step(u_trial,
                                  theta_trial,
                                  theta_trial_per_well,
                                  phi_trial,
                                  step_type)
                print("\nSTEP ACCEPTED")
                print("step_type =", step_type)
                print("new theta_k =", self.theta_k)
                print("new phi_k   =", self.phi_k)
                print("new Delta   =", self.Delta)
            else:
                print("\nSTEP REJECTED")
                print("Delta old =", self.Delta)
                self.Delta *= self.gamma_c
                print("Delta new =", self.Delta)
            k+=1

        return {
            "u_opt": self.u_k,
            "history": self.history
        }

    #################################################################################################################
    ################################# EVALUATE SURROGATES WITH JAC ##################################################
    def _split_u_by_well(self,u_k):
        """
        Split stacked field-level control into a list of per-well controls
        Input:
            u shape = (N*nu,)
        Output:
            [u_0, u_1, ..., u_{N-1}], each shape = (nu,)
        """
        u = np.array(u_k, dtype=float).reshape((-1,))
        assert u.shape == (self.N * self.nu,), \
            f"u has shape {u.shape}, expected {(self.N * self.nu,)}"

        U_list = []
        for j in range(self.N):
            start = j * self.nu
            end = (j + 1) * self.nu
            U_list.append(u[start:end].copy())
        return U_list


    def _eval_surrogates_with_jac(self, u_k):
        """
        Evaluate each well surrogate independently at the current stacked iterate.

        Returns
        -------
        z_surr_k_list : list of np.ndarray
            One output vector per well
        J_surr_k_list : list of np.ndarray
            One local Jacobian per well, shape (nz_j, nu)
        """
        u_k_list=self._split_u_by_well(u_k)

        z_surr_k_list=[]
        J_surr_k_list=[]

        for j, well_name in enumerate(self.well_names):
            F_u2z_j=self.F_u2z_models[j]

            u_j=u_k_list[j]
            u_sym=ca.MX.sym(f"u_{well_name}",self.nu)


            z_j=F_u2z_j(u=u_sym)["z"]
            J_j=ca.jacobian(z_j,u_sym)

            F_eval=ca.Function(f"F_eval_surr_{well_name}",
                                [u_sym],
                                [z_j,J_j],
                                ["u"],
                                ["z","J"]
            )
            out=F_eval(u=u_j)
            z_val = np.array(out["z"]).reshape((-1,))
            J_val = np.array(out["J"])

            z_surr_k_list.append(z_val)
            J_surr_k_list.append(J_val)

        return z_surr_k_list, J_surr_k_list

    def _eval_plant_models(self,
                           u_k):
        """
        Multi-well rigorous plant API.

        Parameters
        ----------
        models : list[dict]
            One DAE model per well.
        u_k : array-like, shape (N*nu,)
            Stacked field control vector.
        y_guess_list : list[array-like]
            One differential-state guess per well.
        z_guess_list : list[array-like]
            One algebraic-state guess per well.
        out_indices : list[int]
            Selected plant outputs to keep in each well z0.

        Returns
        -------
        dict with keys:
            z0_list     : list of ndarray
            J_list      : list of ndarray
            y_star_list : list of ndarray
            z_star_list : list of ndarray
        """
        u_k = np.array(u_k, dtype=float).reshape((-1,))

        # Outputs to build.
        z0_list = []
        J_list = []
        y_star_list = []
        z_star_list = []

        for j, well_name in enumerate(self.well_names):
            model=self.rigorous_models[j]
            well_data = self.wells[well_name]

            # Slice the stacked field control to obtain the local well control
            start=j*self.nu
            end=(j+1)*self.nu
            u_j=u_k[start:end]

            y_guess=well_data["y_guess_rig"]
            z_guess=well_data["z_guess_rig"]

            res_j=self.plant_eval(
                model=model,
                u_k=u_j,
                y_guess=y_guess,
                z_guess=z_guess,
                out_names=self.out_names
            )

            # Store results per well.
            z0_list.append(res_j["z0"])
            J_list.append(res_j["J"])
            y_star_list.append(res_j["y_star"])
            z_star_list.append(res_j["z_star"])

        return z0_list,J_list


    def _build_corrected_models(self,
                                u_k,
                                z_surr_list,
                                J_surr_list,
                                z_p_list,
                                J_p_list):
        """
        Build one corrected first-order model per well:

            z_corr_j(u_j) = z_surr_j(u_j) + c_j + C_j (u_j - u_{j,k})

        where
            c_j = z_plant_j(u_{j,k}) - z_surr_j(u_{j,k})
            C_j = J_plant_j(u_{j,k}) - J_surr_j(u_{j,k})
        """
        u_k_list=self._split_u_by_well(u_k)
        F_corr_models=[]

        for j, well_name in enumerate(self.well_names):
            print(f"Correcting models... for well {well_name}")

            F_u2z_j=self.F_u2z_models[j]

            u_k_j = u_k_list[j]
            print(f"u_k_j = {u_k_j}")
            print(f"u_k_h shape:{u_k_j.shape}")

            z_s_j = np.array(z_surr_list[j]).reshape((-1,))
            J_s_j = np.array(J_surr_list[j])
            z_p_j = np.array(z_p_list[j]).reshape((-1,))
            J_p_j = np.array(J_p_list[j])

            # print("\n--- Surrogate at u_k_j ---")
            # print("z_s_j =", z_s_j)
            # print("z_s_j shape =", z_s_j.shape)
            # print("J_s_j =")
            # print(J_s_j)
            # print("J_s_j shape =", J_s_j.shape)
            #
            # print("\n--- Plant at u_k_j ---")
            # print("z_p_j =", z_p_j)
            # print("z_p_j shape =", z_p_j.shape)
            # print("J_p_j =")
            # print(J_p_j)
            # print("J_p_j shape =", J_p_j.shape)



            c_j = z_p_j - z_s_j
            C_j = J_p_j - J_s_j

            # print("\n--- Correction terms ---")
            # print("c_j = z_p_j - z_s_j =")
            # print(c_j)
            # print("||c_j||_2 =", np.linalg.norm(c_j))
            #
            # print("\nC_j = J_p_j - J_s_j =")
            # print(C_j)
            # print("||C_j||_F =", np.linalg.norm(C_j))

            u_j = ca.MX.sym(f"u_corr_{well_name}", self.nu)
            z_surr_j = F_u2z_j(u=u_j)["z"]

            du_j = u_j - ca.DM(u_k_j)

            # print("\n--- Local displacement model ---")
            # print("du_j = u_j - u_k_j")
            # print("u_k_j DM =")
            # print(ca.DM(u_k_j))

            z_corr_j = z_surr_j + ca.DM(c_j) + ca.DM(C_j) @ du_j

            F_corr_j = ca.Function(
                f"F_corr_{well_name}",
                [u_j],
                [z_corr_j],
                ["u"],
                ["z"]
            )

            # --------------------------------------------------
            # 5) CHECK interpolation at u_k_j
            # --------------------------------------------------
            z_corr_at_uk = np.array(F_corr_j(u=u_k_j)["z"], dtype=float).reshape((-1,))

            # print("\n--- Interpolation check at u_k_j ---")
            # print("z_corr_j(u_k_j) =", z_corr_at_uk)
            # print("z_p_j           =", z_p_j)
            # print("z_corr_j(u_k_j) - z_p_j =")
            # print(z_corr_at_uk - z_p_j)
            # print("||z_corr_j(u_k_j) - z_p_j||_2 =", np.linalg.norm(z_corr_at_uk - z_p_j))

            # --------------------------------------------------
            # 6) CHECK corrected Jacobian at u_k_j
            # --------------------------------------------------
            J_corr_sym = ca.jacobian(z_corr_j, u_j)
            F_corr_eval = ca.Function(
                f"F_corr_eval_{well_name}",
                [u_j],
                [z_corr_j, J_corr_sym],
                ["u"],
                ["z", "J"]
            )

            corr_eval = F_corr_eval(u=u_k_j)
            z_corr_eval = np.array(corr_eval["z"], dtype=float).reshape((-1,))
            J_corr_eval = np.array(corr_eval["J"], dtype=float)

            # print("\n--- Jacobian consistency check at u_k_j ---")
            # print("J_corr_j(u_k_j) =")
            # print(J_corr_eval)
            # print("J_p_j =")
            # print(J_p_j)
            # print("J_corr_j(u_k_j) - J_p_j =")
            # print(J_corr_eval - J_p_j)
            # print("||J_corr_j(u_k_j) - J_p_j||_F =", np.linalg.norm(J_corr_eval - J_p_j))

            F_corr_models.append(F_corr_j)

        return F_corr_models

    def _eval_corrected_models(self,F_corr_models,u):
        """
        Evaluate corrected well models at stacked control vector u
        """
        u_list=self._split_u_by_well(u)
        z_corr_list=[]

        for j in range(self.N):
            z_j=np.array(F_corr_models[j](u=u_list[j])["z"]).reshape((-1,))
            z_corr_list.append(z_j)
        return z_corr_list

    def _compute_phi(self, z_plant_list):
        """
        Field objective used by the filter:
            phi = - total oil
        Assumes w_o_out is index 7 in each well output vector.
        """
        total_w_o = 0.0
        for z_j in z_plant_list:
            total_w_o += float(z_j[7])
        return -total_w_o

    def _compute_theta(self, z_plant_list, z_model_list):
        """
        Model mismatch measure for the filter.
        Here we use the Euclidean norm of the stacked mismatch vector.
        """
        err_blocks = []
        theta_per_well=[]

        for z_p_j, z_m_j in zip(z_plant_list, z_model_list):
            err_j = np.array(z_p_j).reshape((-1,)) - np.array(z_m_j).reshape((-1,))
            err_blocks.append(err_j)
            theta_per_well.append(float(np.linalg.norm(err_j, ord=2)))

        err = np.concatenate(err_blocks) if err_blocks else np.array([0.0])
        theta_total=float(np.linalg.norm(err, ord=2))

        return theta_total, theta_per_well

    def _solve_tr_subproblem(self,
                             F_corr_models):

        # ---------------------
        # 1) Decision variables
        # ---------------------
        U=[]
        Xblocks=[]

        for j in range(self.N):
            u_j=ca.MX.sym(f"u_j{j}",self.nu)
            U.append(u_j)
            Xblocks.append(u_j)

        x=ca.vertcat(*Xblocks)

        # ---------------------
        # 2) Constraints and objective
        # ---------------------

        g_list, lbg, ubg = [], [], []

        total_w_o = 0
        total_w_w = 0
        total_w_g_inj = 0
        total_w_g_res = 0
        total_w_l_out = 0

        for j, well_name in enumerate(self.well_names):
            model=F_corr_models[j]
            well_data = self.wells[well_name]

            z_j=model(u=U[j])["z"]
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
            }

            # totals
            total_w_o += outD["w_o_out"]
            total_w_w += outD["w_w_out"]
            total_w_g_inj += outD["w_G_inj"]
            total_w_g_res += outD["w_G_res"]
            total_w_l_out += outD["w_L_out"]

            # ---------------------
            # 2.1) Per-well constraints
            # ---------------------
            if not self.unconstrained_well:
                # Minimum bottom-hole pressure
                g_list.append(outD["P_bh_bar"])
                lbg.append(self.P_min_bh)
                ubg.append(self.INF)

                # Maximum tubing-bottom pressure
                g_list.append(outD["P_tb_b_bar"])
                lbg.append(0.0)
                ubg.append(self.P_max_tb_b)

            # ---------------------
            # 2.1) Per-well constraints
            # ---------------------
            if self.enforce_stable:
                coef_i=well_data["coeff_stability"]
                u1_j=U[j][0]
                u2_j=U[j][1]
                b_i=polyval_casadi(u1_j,coef_i)

                g_list.append(b_i - u2_j)
                lbg.append(-self.INF)
                ubg.append(0.0)

        # ---------------------
        # 2.2) Objective
        # ---------------------
        obj = -total_w_o

        # ---------------------
        # 2.3) Coupling constraints
        # ---------------------
        if not self.unconstrained_platform:
            if self.G_available is not None:
                g_list.append(total_w_g_inj)
                lbg.append(0.0)
                ubg.append(self.G_available)

            if self.G_max_export is not None:
                g_list.append(total_w_g_res)
                lbg.append(0.0)
                ubg.append(self.G_max_export)

            if self.W_max is not None:
                g_list.append(total_w_w)
                lbg.append(0.0)
                ubg.append(self.W_max)

            if self.L_max is not None:
                g_list.append(total_w_l_out)
                lbg.append(0.0)
                ubg.append(self.L_max)

        # ---------------------
        # 2.4) Trust-region constraint
        # ---------------------

        x_k = ca.DM(self.u_k).reshape((-1, 1))

        dx = x - x_k
        g_list.append(ca.dot(dx, dx))
        lbg.append(-self.INF)
        ubg.append(self.Delta ** 2)

        # ---------------------
        # 2.5) Bounds and initial guess
        # ---------------------
        lbx_list = []
        ubx_list = []

        for _ in range(self.N):
            lbx_list += list(self.u_lb)
            ubx_list += list(self.u_ub)

        lbx = ca.DM(lbx_list).reshape((-1, 1))
        ubx = ca.DM(ubx_list).reshape((-1, 1))
        x0 = x_k
        g=ca.vertcat(*g_list) if g_list else ca.MX()

        # ---------------------
        # 3) Solve NLP
        # ---------------------

        nlp = {"x": x, "f": obj, "g": g}
        solver = ca.nlpsol("trsp_solver", "ipopt", nlp, self.ipopt_opts)

        sol=solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
        )

        # ---------------------
        # 4) Post-process solution
        # ---------------------
        x_star = np.array(sol["x"]).reshape((-1,))
        g_star = np.array(sol["g"]).reshape((-1,)) if "g" in sol else None

        # Split by well
        u_star_list = []
        z_star_list = []

        total_w_o_star = 0.0
        total_w_w_star = 0.0
        total_w_g_inj_star = 0.0
        total_w_g_res_star = 0.0
        total_w_l_out_star = 0.0

        for j, model in enumerate(F_corr_models):
            start = j * self.nu
            end = (j + 1) * self.nu

            u_j_star = x_star[start:end]
            z_j_star = np.array(model(u=u_j_star)["z"]).reshape((-1,))

            u_star_list.append(u_j_star)
            z_star_list.append(z_j_star)

            total_w_o_star += float(z_j_star[7])
            total_w_w_star += float(z_j_star[6])
            total_w_g_inj_star += float(z_j_star[2])
            total_w_g_res_star += float(z_j_star[5])
            total_w_l_out_star += float(z_j_star[6] + z_j_star[7])

        # ---------------------
        # 5) Rich per-well dictionary
        # ---------------------
        wells_solutions = []
        for j in range(self.N):
            z_j = z_star_list[j]
            wells_solutions.append({
                "well_index": j,
                "well_name": self.well_names[j] if hasattr(self, "well_names") else f"well_{j}",
                "u_star": u_star_list[j],
                "P_bh_bar": float(z_j[0]),
                "P_tb_b_bar": float(z_j[1]),
                "w_G_inj": float(z_j[2]),
                "w_res": float(z_j[3]),
                "w_L_res": float(z_j[4]),
                "w_G_res": float(z_j[5]),
                "w_w_out": float(z_j[6]),
                "w_o_out": float(z_j[7]),
                "w_L_out": float(z_j[6] + z_j[7]),
            })

        # ---------------------
        # 6) Trust-region diagnostics
        # ---------------------
        dx_star = x_star - np.array(x_k).reshape((-1,))
        tr_norm_sq = float(np.dot(dx_star, dx_star))
        tr_radius_sq = float(self.Delta ** 2)
        tr_active = tr_norm_sq >= 0.999 * tr_radius_sq

        # ---------------------
        # 7) Return
        # ---------------------
        return {
            "success": bool(solver.stats().get("success", False)),
            "stats": solver.stats(),

            "x_star": x_star,
            "u_star_list": u_star_list,
            "z_star_list": z_star_list,

            "objective_value": -total_w_o_star,
            "total_w_o": total_w_o_star,
            "total_w_w": total_w_w_star,
            "total_w_g_inj": total_w_g_inj_star,
            "total_w_g_res": total_w_g_res_star,
            "total_w_l_out": total_w_l_out_star,

            "tr_norm_sq": tr_norm_sq,
            "tr_radius_sq": tr_radius_sq,
            "tr_active": tr_active,

            "g_star": g_star,
            "wells": wells_solutions,
        }
    # =========================================================
    # FILTER
    # =========================================================

    def _filter_accept(self,
                       theta,
                       phi):

        for e in self.filter_list:
            cond_theta = theta <= (1 - self.gamma_theta) * e["theta"]
            cond_phi   = phi <= e["phi"] - self.gamma_f * e["theta"]

            if not (cond_theta or cond_phi):
                return False

        return True

    def _filter_update(self, theta, phi):
        self.filter_list.append({"theta": theta, "phi": phi})

    # =========================================================
    # STEP HANDLING
    # =========================================================

    def _accept_step(self,
                     u_trial,
                     theta,
                     theta_per_well,
                     phi,
                     step_type):

        self.history.append({
            "u": self.u_k.copy(),
            "u_trial": u_trial.copy(),
            "Delta": self.Delta,
            "theta": theta,
            "theta_per_well": theta_per_well.copy(),
            "phi": phi,
            "type": step_type
        })

        self.u_k = u_trial
        self.theta_k = theta
        self.phi_k = phi


    #
    #
    # def filter_accept(theta_trial, phi_trial, filter_list, gamma_theta=0.01, gamma_phi=0.01):
    #     """
    #     Return True if (theta_trial, phi_trial) is acceptable to the filter.
    #     filter_list is a list of dicts: [{"theta":..., "phi":...}, ...]
    #     """
    #     theta_trial = float(theta_trial)
    #     phi_trial = float(phi_trial)
    #
    #     for e in filter_list:
    #         theta_i = float(e["theta"])
    #         phi_i   = float(e["phi"])
    #
    #         cond_theta = theta_trial <= (1.0 - gamma_theta) * theta_i
    #         cond_phi   = phi_trial   <= phi_i - gamma_phi * theta_i
    #
    #         # must satisfy at least one condition for each filter entry
    #         if not (cond_theta or cond_phi):
    #             return False
    #
    #     return True
    #
    # def filter_update(theta_new, phi_new, filter_list, prune=False):
    #     """
    #     Add (theta_new, phi_new) to the filter.
    #     Optionally prune dominated entries.
    #
    #     Dominance rule (minimize both):
    #       (theta_a, phi_a) dominates (theta_b, phi_b) if:
    #         theta_a <= theta_b and phi_a <= phi_b, with at least one strict.
    #     """
    #     theta_new = float(theta_new)
    #     phi_new = float(phi_new)
    #
    #     filter_list.append({"theta": theta_new, "phi": phi_new})
    #
    #     if not prune:
    #         return filter_list
    #
    #     # prune dominated points
    #     kept = []
    #     for i, a in enumerate(filter_list):
    #         dominated = False
    #         for j, b in enumerate(filter_list):
    #             if i == j:
    #                 continue
    #             if (b["theta"] <= a["theta"] and b["phi"] <= a["phi"] and
    #                 (b["theta"] < a["theta"] or b["phi"] < a["phi"])):
    #                 dominated = True
    #                 break
    #         if not dominated:
    #             kept.append(a)
    #
    #     # also remove near-duplicates
    #     uniq = []
    #     for e in kept:
    #         if not any(abs(e["theta"]-q["theta"]) < 1e-12 and abs(e["phi"]-q["phi"]) < 1e-12 for q in uniq):
    #             uniq.append(e)
    #
    #     return uniq

    #
    # #####################################################################
    # # 1 - SOLVER INITIALIZATION. RUNS JUST ONCE
    # #####################################################################
    #
    # print("\n============= SOLVER INITIALIZATION START =============\n")
    #
    #
    #
    # k=0
    # filter_list=[]
    # u_k=u_k_init
    # y_guess_rig=y_guess_rig_init
    # z_guess_rig=z_guess_rig_init
    # Delta=Delta_init
    # min_obj=np.inf
    # max_infeasibility=-np.inf
    # theta_tol=1e-6
    # theta_k=None
    # phi_plant_k=None
    #
    # max_iter=100
    #
    #
    #
    # while k<max_iter:
    #
    #
    #
    #     # res = F_rk(u=u_k)
    #     # z_k = np.array(res["z"]).reshape((-1,))
    #     # J_k = np.array(res["J"])
    #
    #     # print("\n----- SURROGATE LOCAL MODEL -----")
    #     # print("r_k(u_k) = [w_o_out, P_bh_bar, P_tb_b_bar]:")
    #     # print(z_k)
    #     #
    #     # print("\n∇r_k(u_k)  (Jacobian dy/du)")
    #     # print(J_k)
    #     #
    #     # print("\nInterpretation:")
    #     # print("Rows  = outputs [w_out,P_bh,P_tb_b]")
    #     # print("Cols  = controls [u1,u2]")
    #     # print("Entry (i,j) = ∂y_i / ∂u_j")
    #     # print("-------------------------------------\n")
    #
    #     simulation_import_api=plant_zeroth_and_first_order(
    #         model=plant_model,
    #         u_k=u_k,
    #         y_guess=y_guess_rig,
    #         z_guess=z_guess_rig
    #     )
    #
    #     y_guess_rig=simulation_import_api["y_star"]
    #     z_guess_rig=simulation_import_api["z_star"]
    #
    #     plant_z_k=simulation_import_api["z0"]
    #     phi_k_current = None if phi_plant_k is None else float(phi_plant_k)
    #     theta_k_current = None if theta_k is None else float(theta_k)
    #
    #     plant_J_k=simulation_import_api["J"]
    #
    #     print("\n----- TRUTH MODEL -----")
    #
    #     print("Plant z0(u_k) = [w_o_out, P_bh_bar, P_tb_b_bar]:")
    #     print(plant_z_k)
    #
    #
    #
    #     print("\nPlant J(u_k) = dz/du:")
    #     print(plant_J_k)
    #
    #     print("\nInterpretation:")
    #     print("Rows  = outputs [w_out,P_bh,P_tb_b]")
    #     print("Cols  = controls [u1,u2]")
    #     print("Entry (i,j) = ∂y_i / ∂u_j")
    #     print("-------------------------------------\n")
    #
    #     u = ca.MX.sym("u", 2)
    #     z_surr = F_u2z(u=u)["z"]  # (3x1)
    #
    #     c_k = plant_z_k - z_k  # (3,)
    #     C_k = plant_J_k - J_k  # (3,2)
    #
    #     du = u - ca.DM(u_k)
    #     z_corr = z_surr + ca.DM(c_k) + ca.DM(C_k) @ du
    #
    #     F_u2z_corr = ca.Function(
    #         f"F_u2z_corr_{k}",
    #         [u],
    #         [z_corr],
    #         ["u"],
    #         ["z"]
    #     )
    #
    #     print("\n--- CORRECTION TERMS at iteration ---")
    #     print("c_k      =", c_k)
    #     print("\nC_k =\n", C_k)
    #     print("------\n")
    #
    #     if theta_k is not None and theta_k<theta_tol:
    #         print("\nAlgorithm converged")
    #         print(u_k)
    #         print(z_plant_trial)
    #         print(z_surr_trial)
    #         u_converged = np.array(u_k, dtype=float).reshape((2,))
    #         return {
    #             "u_converged": u_converged,
    #             "history": history,
    #         }
    #     print("------\n")
    #
    #     print("\n--- Solving TRSP ---")
    #
    #     ###############################################################
    #     # 3 - TRUST REGION SUBPROBLEM
    #     ###############################################################
    #     # F_u2z_corr is your corrected CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
    #     res = solve_trust_region_subproblem(
    #         F_u2z=F_u2z_corr,
    #         u_k=u_k,
    #         Delta=Delta,
    #         u_min=u_min,
    #         u_max=u_max,
    #         P_min_bh=P_min_bh,
    #         P_max_tb_b=P_max_tb_b
    #     )
    # #
    #     print("Solver success:", res["stats"]["success"])
    #     print("u* =", res["u_star"])
    #     print("m_o_out* =", res["m_o_out"])
    #     print("P_bh* =", res["P_bh"])
    #     print("P_tb_b* =", res["P_tb_b"])
    # #
    #     u_k=np.array(u_k,dtype=float).reshape((2,))
    #     z_k = np.array(F_u2z(u=u_k)["z"]).reshape((-1,))
    #
    #     u_star=np.array(res["u_star"],dtype=float).reshape((2,))
    #
    #     m_os, Pbh_s, Ptb_s = res["m_o_out"], res["P_bh"], res["P_tb_b"]
    #     print("m_os =", m_os)
    #     print("Pbh_s =", Pbh_s)
    #     print("Ptb_s =", Ptb_s)
    #
    #     m_ok,Pbh_k,Ptb_k=z_k[0],z_k[1],z_k[2]
    #
    #     #So what was the step size?
    #     step = u_star - u_k
    #     step_norm = float(np.linalg.norm(step))
    #     hit_TR = step_norm >= 0.999 * float(Delta)
    #
    #     # Constraint slacks (>=0 means satisfied, =0 active)
    #     slack_Pbh = float(Pbh_s - P_min_bh)  # >= 0
    #     slack_Ptb = float(P_max_tb_b - Ptb_s)  # >= 0
    #
    #     u1, u2 = float(u_star[0]), float(u_star[1])
    #     b_hat = -0.3268 * u1 * u1 + 0.5116 * u1 + 0.01914
    #     slack_stab = float(u2 - b_hat)  # >= 0
    #
    #     # Improvement
    #     dm = float(m_os - m_ok)
    #
    #     print("\n--- TRUST-REGION SUBPROBLEM STATISTICS ---")
    #     print("success:", res["success"], res["stats"].get("return_status", ""))
    #     print(f"u_k    = {u_k}")
    #     print(f"u_star = {u_star}")
    #     print(f"step   = {step} | ||step||={step_norm:.6f}  (Delta={Delta})  hit_TR={hit_TR}")
    #
    #     print("\nObjective:")
    #     print(f"m_o_out(u_k)    = {m_ok:.6f}")
    #     print(f"m_o_out(u_star) = {m_os:.6f}")
    #     print(f"Δm_o_out        = {dm:.6f}")
    #
    #     print("\nConstraints at u_star (slack >= 0 means satisfied):")
    #     print(f"P_bh     = {Pbh_s:.6f}  (>= {P_min_bh})  slack = {slack_Pbh:.6f}")
    #     print(f"P_tb_b   = {Ptb_s:.6f}  (<= {P_max_tb_b})  slack = {slack_Ptb:.6f}")
    #     print(f"stability u2-b_hat = {slack_stab:.6f}  (>= 0)")
    #     print("------------------------------------------------------------\n")
    #
    #     ###############################################################
    #     # 4 - FILTER
    #     ###############################################################
    #     print("\n--- FILTER CHECK ---")
    #     u_trial=u_star
    #
    #     z_surr_trial = np.array(F_u2z_corr(u=u_trial)["z"]).reshape((-1,))
    #     print(f"Z_surr_trial =",z_surr_trial)
    #
    #     plant_res_trial=plant_zeroth_and_first_order(
    #         model=plant_model,
    #         u_k=u_trial,
    #         y_guess=y_guess_rig,
    #         z_guess=z_guess_rig)
    #
    #     y_guess_rig = plant_res_trial["y_star"]
    #     z_guess_rig = plant_res_trial["z_star"]
    #
    #     z_plant_trial=plant_res_trial["z0"]
    #     print(f"z_plant_trial =",z_plant_trial)
    #
    #     phi_plant_trial=-float(z_plant_trial[0])
    #
    #     print(f"The objective we are are minimizing is {phi_plant_trial:.6f} at trial whereas our minimum is {min_obj}")
    #     if phi_plant_trial<min_obj:
    #        min_obj=phi_plant_trial
    #
    #     theta_vec = np.abs(z_plant_trial - z_surr_trial)  # (3,)
    #     theta_trial = float(np.linalg.norm(theta_vec, ord=2))  # Euclidean
    #
    #     print("u_trial =", u_trial)
    #     print("theta_trial =", theta_trial)
    #     print("phi_trial (plant) =", phi_plant_trial)
    #
    #     if k==0:
    #         accepted=True
    #
    #         history["k"].append(k)
    #         history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
    #         history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
    #         history["Delta"].append(float(Delta))
    #         history["theta_k"].append(theta_k_current)
    #         history["theta_trial"].append(float(theta_trial))
    #         history["phi_k"].append(phi_k_current)
    #         history["phi_trial"].append(float(phi_plant_trial))
    #         history["accepted"].append(True)
    #         history["step_type"].append("init")
    #
    #         u_k=u_trial.tolist()
    #         theta_k=theta_trial
    #         phi_plant_k = phi_plant_trial
    #         k = k + 1
    #         filter_list = filter_update(theta_k, phi_plant_k, filter_list, prune=False)
    #         continue
    #     else:
    #         accepted = filter_accept(
    #             theta_trial=theta_trial,
    #             phi_trial=phi_plant_trial,  # usually use PLANT values in the filter
    #             filter_list=filter_list
    #         )
    #     print("Accepted by filter?", accepted)
    #     if accepted:
    #         print(f"        --- Switching Condition ---")
    #         if phi_plant_k-phi_plant_trial>=0.01*(theta_k**0.9):
    #             print(f"        This is an f-type step")
    #             print(f"        That means that filter is not updated and trust-region increases")
    #             history["k"].append(k)
    #             history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
    #             history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
    #             history["Delta"].append(float(Delta))
    #             history["theta_k"].append(float(theta_k))
    #             history["theta_trial"].append(float(theta_trial))
    #             history["phi_k"].append(float(phi_plant_k))
    #             history["phi_trial"].append(float(phi_plant_trial))
    #             history["accepted"].append(True)
    #             history["step_type"].append("f-type")
    #
    #             Delta=Delta*2
    #             u_k = u_trial.tolist()
    #             theta_k = theta_trial
    #             phi_plant_k = phi_plant_trial
    #             k = k + 1
    #             continue
    #         else:
    #             print(f"        This is an theta-type step")
    #             print(f"        That means that filter is updated and trust-region changes according to the ratio test")
    #             history["k"].append(k)
    #             history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
    #             history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
    #             history["Delta"].append(float(Delta))
    #             history["theta_k"].append(float(theta_k))
    #             history["theta_trial"].append(float(theta_trial))
    #             history["phi_k"].append(float(phi_plant_k))
    #             history["phi_trial"].append(float(phi_plant_trial))
    #             history["accepted"].append(True)
    #             history["step_type"].append("theta-type")
    #             u_k = u_trial.tolist()
    #             rho_k=1-(theta_trial/theta_k)
    #             if rho_k<0.4:
    #                 Delta=Delta*0.5
    #             elif rho_k<0.8:
    #                 Delta=Delta
    #             else:
    #                 Delta=Delta*2.0
    #
    #             filter_list = filter_update(theta_k, phi_plant_k, filter_list, prune=False)
    #             phi_plant_k = phi_plant_trial
    #             theta_k = theta_trial
    #             k = k + 1
    #
    #     else:
    #         # reject (typical action: shrink trust region Delta)
    #         print("Step failed")
    #         history["k"].append(k)
    #         history["u_k"].append(np.array(u_k, dtype=float).reshape((2,)).copy())
    #         history["u_trial"].append(np.array(u_trial, dtype=float).reshape((2,)).copy())
    #         history["Delta"].append(float(Delta))
    #         history["theta_k"].append(theta_k_current)
    #         history["theta_trial"].append(float(theta_trial))
    #         history["phi_k"].append(phi_k_current)
    #         history["phi_trial"].append(float(phi_plant_trial))
    #         history["accepted"].append(False)
    #         history["step_type"].append("reject")
    #
    #         Delta *= 0.5
    #         k=k+1
    # return {
    #     "u_converged": None,
    #     "history": history,
    # }