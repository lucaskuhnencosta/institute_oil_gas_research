import numpy as np
from optimization.production_optimizer import polyval_casadi
np.set_printoptions(suppress=True, precision=6)
from utilities.block_builders import *
from optimization.api_interface_simulator import plant_zeroth_and_first_order

class SurrogateBasedOptimization:
    def __init__(self,
                 config,
                 wells,
                 rigorous_models,
                 F_u2z_models,
                 refinement,
                 u_guess=None,
                 total_wo=None):

        self.theta_compat_tol=1e-8

        # -------------------------------
        # Refinement mode?
        # -------------------------------
        self.refinement=refinement

        # -------------------------------
        # Only for refinement
        # -------------------------------
        self.total_wo=total_wo
        self.oil_allowance=config["oil_allowance"]

        # -------------------------------
        # Models and wells passed to the solver
        # -------------------------------
        self.rigorous_models=rigorous_models
        self.F_u2z_models=F_u2z_models
        self.wells=wells

        # -------------------------------
        # Number of controls is always 2 for this thesis, INF is always 10e20
        # -------------------------------
        self.nu=2
        self.INF=10e20

        # -------------------------------
        # Name and number of wells
        # -------------------------------
        self.well_names = list(wells.keys())
        self.N = len(self.well_names)

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
        self.unconstrained_platform=config["unconstrained_platform"]
        self.enforce_stable=config["enforce_stable"]

        # -------------------------------
        # Surrogate-based optimization parameters
        # -------------------------------
        self.max_iter=config["max_iter"]
        self.theta_mode=config["theta_mode"] # This is to be extinct
        self.Delta=config["Delta"]
        self.gamma_c=config["gamma_c"] # Decrease of the trust region
        self.gamma_e=config["gamma_e"] # Increase of the trust region
        self.gamma_f=config["gamma_f"]
        self.gamma_theta=config["gamma_theta"]
        self.k_theta=config["k_theta"]
        self.gamma_s=config["gamma_s"]
        self.eta_1=config["eta_1"]
        self.eta_2=config["eta_2"]
        self.theta_tol=config["theta_tol"]

        # -------------------------------
        # Here we import the only external API
        # -------------------------------
        self.plant_eval=plant_zeroth_and_first_order

        # -------------------------------
        # You can guess or the guess will come from config
        # -------------------------------
        if u_guess is not None:
            self.u0=u_guess.reshape(-1)
        else:
            self.u0= config["u_guess_list"].reshape(-1)

        # -------------------------------
        # Initialize u_k, theta_k, phi_k, and filter
        # -------------------------------
        self.u_k=np.array(self.u0, dtype=float)
        self.theta_k=None
        self.phi_k=None
        self.filter_list=[]

        # -------------------------------
        # Upper and lower bounds from u need to come from config
        # -------------------------------
        self.u_lb=config["u_lb"]
        self.u_ub=config["u_ub"]

        # -------------------------------
        # History is fundamental for post-processing
        # -------------------------------
        self.history=[]

        # -------------------------------
        # out_names
        # -------------------------------
        self.out_names=config["out_names"]

        # -------------------------------
        # solver opts are changed here directly
        # -------------------------------
        self.ipopt_opts={
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 6000,
        "ipopt.tol": 1e-10,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        }

    def run_phase_zero(self):
        print("\n=====================================================")
        print("=============== PHASE ZERO =========================")
        print("=====================================================")

        for k0 in range(self.max_iter):
            print(f"\n--- PHASE ZERO ITERATION {k0} ---")
            print("u_k =", self.u_k)

            z_surr_k_list, J_surr_k_list = self._eval_surrogates_with_jac(self.u_k)
            z_plant_k_list, J_plant_k_list = self._eval_plant_models(self.u_k)


            theta_comp_k, theta_comp_details_k = self._compute_theta_constraint_infeasibility(
                z_plant_list=z_plant_k_list,
                z_model_list=None,
                u=self.u_k,
                include_model_mismatch=False
            )

            print("theta_compatibility =", theta_comp_k)

            if theta_comp_k <= self.theta_compat_tol:
                print("\nPHASE ZERO CONVERGED")
                print("theta_compatibility =", theta_comp_k)
                return True

            F_corr_models, debug_corrected_models = self._build_corrected_models(
                self.u_k,
                z_surr_k_list,
                J_surr_k_list,
                z_plant_k_list,
                J_plant_k_list
            )


            phase0 = self._solve_phase_zero_subproblem(F_corr_models)
            # print(phase0)

            if not phase0["success"]:
                print("Phase-zero TRSP failed.")
                print("status =", phase0["stats"].get("return_status", "unknown"))
                self.Delta *= self.gamma_e
                continue

            u_trial = phase0["x_star"]

            try:
                z_plant_trial_list, _ = self._eval_plant_models(u_trial)
            except RuntimeError as e:
                print("Plant evaluation failed at phase-zero trial.")
                print("u_trial =", u_trial)
                print("error =", e)
                self.Delta *= self.gamma_c
                continue

            theta_comp_trial, theta_comp_details_trial = self._compute_theta_constraint_infeasibility(
                z_plant_list=z_plant_trial_list,
                z_model_list=None,
                u=u_trial,
                include_model_mismatch=False
            )

            print("theta_comp_trial =", theta_comp_trial)
            print("dtheta_comp      =", theta_comp_trial - theta_comp_k)

            if theta_comp_trial< theta_comp_k:
                self.u_k = u_trial
                self.Delta *= self.gamma_e
            else:
                self.Delta *= self.gamma_c

        print("\nPHASE ZERO FAILED TO REACH COMPATIBILITY")
        return False



    # =========================================================
    # PUBLIC API
    # =========================================================
    def solve(self):

        phase_zero_success=self.run_phase_zero()
        if not phase_zero_success:
            return {
                "success": False,
                "status": "phase_zero_failed",
                "history": self.history,
            }

        k=0
        while k<self.max_iter:
            print("\n=====================================================")
            print(f"=============== ITERATION {k} ======================")
            print("=====================================================")
            print("u_k =", self.u_k)
            """
            u_k is a numpy.ndarray with 2N elements
            Shape (2N,)
            """
            # -------------------------------
            # 1. Evaluate surrogate + plant at current accepted point
            # -------------------------------
            z_surr_k_list, J_surr_k_list = self._eval_surrogates_with_jac(self.u_k)
            z_plant_k_list, J_plant_k_list = self._eval_plant_models(self.u_k)
            """
            z_surr_k_list is a list with 8 elements of r_k(u_k) for EACH well N
            r_k(u_k):[P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]
            
            J_surr_k_list is a list with 8x2 elements each. The derivative of each output with respect to u
            ∇r_k(u_k): [dP_bh_bar/du1, dP_bh_bar/du2] ...
            
            Same applies to plant!
            """
            # -------------------------------
            # 2. The first iteration must initialize the filter
            # -------------------------------
            if k==0 and self.theta_k is None and self.phi_k is None:
                # We just need to compute this here. For the next iterations, it will be computed as of the trials
                # -------------------------------
                # 2.1. Compute Phi
                # -------------------------------
                phi_k=self._compute_phi(z_plant_k_list)
                if self.refinement:
                    print("Starting at production of gas:",phi_k)
                # -------------------------------
                # 2.2 Compute theta
                # -------------------------------
                theta_k,theta_k_details=self._compute_theta(z_plant_k_list,
                                                             z_surr_k_list,
                                                             u=self.u_k)
                self.theta_k=theta_k
                self.phi_k=phi_k

                # -------------------------------
                # 2.3 Initialize the filter
                # -------------------------------
                self._filter_update(theta_k,phi_k)

                self.history.append({
                    "phase": "refinement" if self.refinement else "oil_maximization",

                    "iteration":k,

                    "z_surr": z_surr_k_list,
                    "J_surr": J_surr_k_list,
                    "z_plant": z_plant_k_list,
                    "J_plant": J_plant_k_list,

                    "debug_corrected": None,

                    "u": self.u_k.copy(),
                    "u_trial": self.u_k.copy(),

                    "Delta": self.Delta,
                    "trsp": None,

                    "accepted": None,
                    "rejected": None,
                    "rejected_reason": None,

                    "theta_k": theta_k,
                    "theta_details": theta_k_details,
                    "phi_k": phi_k,
                    "type": "init",

                    "filter_list": self.filter_list,
                })

                print("\n--- FILTER INITIALIZATION ---")
                print("u0      =", self.u_k)
                print("phi_0   =", phi_k)
                print("theta_0 =", theta_k)

            # -------------------------------
            # 3. Convergence test at current accepted point
            # -------------------------------
            if self.theta_k <= self.theta_tol:
                if not self.refinement:
                    oil_total = -self.phi_k
                    print("\n=== CONVERGENCE ACHIEVED ===")
                    print("theta_k=", self.theta_k)
                    print("theta_tol=",self.theta_tol)
                    print("u_converged=",self.u_k)
                    print("oil_total =", oil_total)
                if self.refinement:
                    gas_injected=self.phi_k
                    print("\n=== CONVERGENCE ACHIEVED ===")
                    print("theta_k=", self.theta_k)
                    print("theta_tol=",self.theta_tol)
                    print("u_converged=",self.u_k)
                    print("gas_injected=",gas_injected)


                return {
                    "success": True,
                    "status": "converged",
                    "u_opt": self.u_k.copy(),
                    "theta_opt": self.theta_k,
                    "phi_opt": self.phi_k,
                    "iterations": k,
                    "history": self.history
                }
            # -------------------------------
            # 4. Build corrected models
            # -------------------------------
            F_corr_models, debug_corrected_models = self._build_corrected_models(
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

            if not trsp["success"]:
                print("\n--- TRSP FAILED ---")
                print("status =", trsp["stats"].get("return_status", "unknown"))
                print("Rejecting TRSP solution and shrinking trust region.")

                self.Delta *= self.gamma_c
                k += 1
                continue

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
            try:
                z_plant_trial_list, _ = self._eval_plant_models(u_trial)
            except RuntimeError as e:
                print("\nPLANT EVALUATION FAILED AT TRIAL POINT")
                print("u_trial =", u_trial)
                print("reason =", e)
                print("Rejecting step and shrinking trust region.")

                self.Delta *= self.gamma_c
                k += 1
                continue

            phi_trial = self._compute_phi(z_plant_trial_list)
            theta_trial,theta_trial_per_well = self._compute_theta(z_plant_trial_list,
                                                                   z_surr_trial_list,
                                                                   u=u_trial)

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
            print(f"\nEvaluating plant model for well {well_name}...")

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
        dubug={}
        F_corr_models=[]

        for j, well_name in enumerate(self.well_names):
            # print(f"\nCorrecting models... for well {well_name}")

            F_u2z_j=self.F_u2z_models[j]

            u_k_j = u_k_list[j]

            z_s_j = np.array(z_surr_list[j]).reshape((-1,))
            J_s_j = np.array(J_surr_list[j])
            z_p_j = np.array(z_p_list[j]).reshape((-1,))
            J_p_j = np.array(J_p_list[j])

            """
            u_k_j is shape (2,) always
            After reshape,
            z_s_j shape =(8,)
            J_s_j shape =(8,2)
            z_p_j shape =(8,)
            J_p_j shape =(8,2)
            """

            c_j = z_p_j - z_s_j
            C_j = J_p_j - J_s_j

            norm_c_j=np.linalg.norm(c_j)
            norm_C_j=np.linalg.norm(C_j)

            """
            c_j is shape (8,)
            C_j is shape (8,2)
            """

            u_j = ca.MX.sym(f"u_corr_{well_name}", self.nu)
            z_surr_j = F_u2z_j(u=u_j)["z"]

            du_j = u_j - ca.DM(u_k_j)
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
            check_zeroth=z_corr_at_uk - z_p_j

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
            J_corr_eval = np.array(corr_eval["J"], dtype=float)
            check_first=J_corr_eval - J_p_j

            F_corr_models.append(F_corr_j)

            debug={
                "u_k": u_k_j,
                "c_j": c_j,
                "C_j": C_j,
                "norm_c_j": norm_c_j,
                "norm_C_j": norm_C_j,
                "check_zeroth": check_zeroth,
                "check_first": check_first,
            }

        return F_corr_models, debug

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
        If we are not in the refinement phase:
            phi = - total oil
        Assumes w_o_out is index 7 in each well output vector.

        If we are in the refinement phase:
            phi= total_w_G_inj
        Assumes w_G_inj is index 2 in each well output vector.
        """
        if not self.refinement:
            total_w_o = 0.0
            for z_j in z_plant_list:
                total_w_o += float(z_j[7])
            return -total_w_o
        else:
            total_w_G_inj=0.0
            for z_j in z_plant_list:
                total_w_G_inj += float(z_j[2])
            return total_w_G_inj

    def _compute_theta(self,
                       z_plant_list,
                       z_model_list,
                       u):
        ### THIS BLOCK WILL BE DECOMISSIONED SOON
        if self.theta_mode=="model_mismatch":
            return self._compute_theta_model_mismatch(z_plant_list,
                                                      z_model_list)
        ####
        elif self.theta_mode=="constraint_infeasibility":
            return self._compute_theta_constraint_infeasibility(z_plant_list=z_plant_list,
                                                                z_model_list=z_model_list,
                                                                u=u,
                                                                include_model_mismatch=True)
            raise ValueError(f"Unknown theta mode: {self.theta_mode}")

    def _positive_part(self, x):
        return max(0.0, float(x))



    def _compute_theta_constraint_infeasibility(self,
                                                z_plant_list,
                                                z_model_list,
                                                u,
                                                include_model_mismatch):
        """
        Constraint infeasibility measure.

        Per-well constraints have weight 1.
        Global/platform constraint block has weight N.

        Pressure residuals are divided by 100.
        Flow residuals are left unscaled.

        Returns
        -------
        theta_total: float
        theta_blocks: dict
            Diagnostic infeasibility values per block
        """

        theta_vector=[]

        theta_details={
            "model_mismatch_per_well":[],
            "well_constraint_per_well":[],
            "platform_constraints":{},
        }

        u_list=self._split_u_by_well(u)
        total_w_o=0.0
        total_w_w=0.0
        total_w_g_inj=0.0
        total_w_g_res=0.0
        total_w_l_out=0.0


        # -------------------------------
        # 1. Model infeasibility
        # ------------------------------



        if include_model_mismatch:
            if z_model_list is None:
                raise ValueError("z_model_list must not be None")

            scale = np.array([
                1 / 100,  # P_bh_bar
                1 / 100,  # P_tb_b_bar
                10.0,  # w_G_inj
                1.0,  # w_res
                1.0,  # w_L_res
                1.0,  # w_G_res
                1.0,  # w_w_out
                1.0,  # w_o_out
            ], dtype=float)

            for z_p_j, z_m_j in zip(z_plant_list, z_model_list):
                err_j = np.array(z_p_j).reshape((-1,)) - np.array(z_m_j).reshape((-1,))
                err_j_scaled = scale * err_j

                theta_model_j = float(np.linalg.norm(err_j_scaled, ord=2))

                theta_vector.append(theta_model_j)
                theta_details["model_mismatch_per_well"].append(theta_model_j)

        # =====================================================
        # 2. Per-well infeasibility blocks
        # =====================================================
        for j, z_j in enumerate(z_plant_list):
            z_j=np.array(z_j).reshape((-1,))
            u_j=u_list[j]

            P_bh_bar = float(z_j[0])
            P_tb_b_bar = float(z_j[1])
            w_G_inj = float(z_j[2])
            w_G_res = float(z_j[5])
            w_w_out = float(z_j[6])
            w_o_out = float(z_j[7])
            w_L_out = w_w_out + w_o_out

            total_w_o += w_o_out
            total_w_w += w_w_out
            total_w_g_inj += w_G_inj
            total_w_g_res += w_G_res
            total_w_l_out += w_L_out

            violations_j = []

            if not self.unconstrained_well:
                # P_bh_bar >= P_min_bh
                violations_j.append(
                    self._positive_part(self.P_min_bh - P_bh_bar) / 100.0
                )

                # P_tb_b_bar <= P_max_tb_b
                violations_j.append(
                    self._positive_part(P_tb_b_bar - self.P_max_tb_b) / 100.0
                )

            if self.enforce_stable:
                well_data = self.wells[self.well_names[j]]
                coef_i = well_data["coeff_stability"]

                u1_j = float(u_j[0])
                u2_j = float(u_j[1])

                b_i = float(np.polyval(coef_i, u1_j))

                violations_j.append(
                    self._positive_part(b_i - u2_j)
                )

            theta_well_j = (
                float(np.linalg.norm(np.array(violations_j), ord=2))
                if violations_j
                else 0.0
            )

            theta_vector.append(theta_well_j)
            theta_details["well_constraint_per_well"].append(theta_well_j)

        # =====================================================
        # 3. Platform constraints
        # =====================================================
        if not self.unconstrained_platform:
            if self.G_available is not None:
                v = self._positive_part(total_w_g_inj - self.G_available)
                theta_vector.append(self.N * v)
                theta_details["platform_constraints"]["G_available"] = v

            if self.G_max_export is not None:
                v = self._positive_part(total_w_g_res - self.G_max_export)
                theta_vector.append(self.N * v)
                theta_details["platform_constraints"]["G_max_export"] = v

            if self.W_max is not None:
                v = self._positive_part(total_w_w - self.W_max)
                theta_vector.append(self.N * v)
                theta_details["platform_constraints"]["W_max"] = v

            if self.L_max is not None:
                v = self._positive_part(total_w_l_out - self.L_max)
                theta_vector.append(self.N * v)
                theta_details["platform_constraints"]["L_max"] = v

        theta_vector = np.array(theta_vector, dtype=float)
        theta_total = float(np.linalg.norm(theta_vector, ord=2))

        theta_details["theta_vector"] = theta_vector
        theta_details["theta_total"] = theta_total

        return theta_total, theta_details


    def _compute_theta_model_mismatch(self,z_plant_list,z_model_list):
        """
        Model mismatch measure for the filter.
        Here we use the Euclidean norm of the stacked mismatch vector.
        """
        scale = np.array([
            1 / 100,  # P_bh_bar
            1 / 100,  # P_tb_b_bar
            10.0,  # w_G_inj
            1.0,  # w_res
            1.0,  # w_L_res
            1.0,  # w_G_res
            1.0,  # w_w_out
            1.0,  # w_o_out
        ],dtype=float)

        err_blocks = []
        theta_per_well=[]

        for z_p_j, z_m_j in zip(z_plant_list, z_model_list):
            err_j = np.array(z_p_j).reshape((-1,)) - np.array(z_m_j).reshape((-1,))

            err_j_scaled = scale * err_j

            err_blocks.append(err_j_scaled)

            theta_per_well.append(float(np.linalg.norm(err_j_scaled, ord=2)))

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
        if not self.refinement:
            obj = -total_w_o
        else:
            obj = total_w_g_inj

        if self.refinement:
            g_list.append(total_w_o)
            lbg.append(self.total_wo-self.oil_allowance)
            ubg.append(self.INF)

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

        # # ---------------------
        # # 6) Trust-region diagnostics
        # # ---------------------
        # x_star = np.array(sol["x"], dtype=float).reshape((-1,))
        # x_k_np = np.array(self.u_k, dtype=float).reshape((-1,))
        #
        # dx_star = x_star - x_k_np
        #
        # # dx_star = x_star - np.array(x_k).reshape((-1,))
        # tr_norm_sq = float(np.dot(dx_star, dx_star))
        # tr_radius_sq = float(self.Delta ** 2)
        # tr_active = tr_norm_sq >= 0.999 * tr_radius_sq

        # ---------------------
        # 6) Trust-region diagnostics
        # ---------------------
        x_star = np.array(sol["x"], dtype=float).reshape((-1,))
        x_k_np = np.array(self.u_k, dtype=float).reshape((-1,))

        dx_star = x_star - x_k_np

        tr_norm_sq = float(np.dot(dx_star, dx_star))
        tr_radius_sq = float(self.Delta ** 2)
        tr_violation = tr_norm_sq - tr_radius_sq

        # Constraint-vector diagnostic
        g_star = np.array(sol["g"], dtype=float).reshape((-1,))
        lbg_np = np.array(lbg, dtype=float).reshape((-1,))
        ubg_np = np.array(ubg, dtype=float).reshape((-1,))

        tr_g_index = len(g_star) - 1
        tr_g_value = float(g_star[tr_g_index])
        tr_lbg = float(lbg_np[tr_g_index])
        tr_ubg = float(ubg_np[tr_g_index])

        print("\n--- TRUST REGION DEBUG ---")
        print("solver success =", solver.stats().get("success", None))
        print("return_status  =", solver.stats().get("return_status", None))

        print("x_star =", x_star)
        print("x_k_np =", x_k_np)
        print("dx_star =", dx_star)

        print("manual ||dx||^2 =", tr_norm_sq)
        print("Delta^2         =", tr_radius_sq)
        print("manual violation =", tr_violation)

        print("len(g_star) =", len(g_star))
        print("TR g index  =", tr_g_index)
        print("TR g value  =", tr_g_value)
        print("TR lbg      =", tr_lbg)
        print("TR ubg      =", tr_ubg)
        print("TR g - ubg  =", tr_g_value - tr_ubg)

        tr_feasible = tr_g_value <= tr_ubg + 1e-6
        tr_active = tr_feasible and tr_g_value >= 0.999 * tr_ubg

        # ---------------------
        # 7) Return
        # ---------------------
        return {
            "success": bool(solver.stats().get("success", False)),
            "stats": solver.stats(),

            "x_star": x_star,
            "u_star_list": u_star_list,
            "z_star_list": z_star_list,

            "objective_value": obj,
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

    def _solve_phase_zero_subproblem(self, F_corr_models):
        """
        Phase-zero compatibility subproblem.

        Minimize smooth slack-based infeasibility of all inequality constraints,
        subject only to:
            - variable bounds
            - trust-region constraint

        This avoids ca.fmax(...) and gives IPOPT a smooth NLP.
        """

        # ---------------------
        # 1) Decision variables
        # ---------------------
        U = []
        Xblocks = []

        for j in range(self.N):
            u_j = ca.MX.sym(f"u_phase0_j{j}", self.nu)
            U.append(u_j)
            Xblocks.append(u_j)

        x = ca.vertcat(*Xblocks)

        # ---------------------
        # 2) Constraints
        # ---------------------
        g_list, lbg, ubg = [], [], []

        slack_vars = []
        slack_lbx = []
        slack_ubx = []
        slack_x0 = []

        total_w_o = 0
        total_w_w = 0
        total_w_g_inj = 0
        total_w_g_res = 0
        total_w_l_out = 0

        # ---------------------
        # 2.1) Per-well constraints with slacks
        # ---------------------
        for j, well_name in enumerate(self.well_names):
            model = F_corr_models[j]
            well_data = self.wells[well_name]

            z_j = model(u=U[j])["z"]

            P_bh_bar = z_j[0]
            P_tb_b_bar = z_j[1]
            w_G_inj = z_j[2]
            w_G_res = z_j[5]
            w_w_out = z_j[6]
            w_o_out = z_j[7]
            w_L_out = w_w_out + w_o_out

            total_w_o += w_o_out
            total_w_w += w_w_out
            total_w_g_inj += w_G_inj
            total_w_g_res += w_G_res
            total_w_l_out += w_L_out

            if not self.unconstrained_well:
                # P_bh_bar >= P_min_bh
                s_pbh = ca.MX.sym(f"s_pbh_{j}")
                slack_vars.append(s_pbh)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(P_bh_bar + 100.0 * s_pbh)
                lbg.append(self.P_min_bh)
                ubg.append(self.INF)

                # P_tb_b_bar <= P_max_tb_b
                s_ptb = ca.MX.sym(f"s_ptb_{j}")
                slack_vars.append(s_ptb)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(P_tb_b_bar - 100.0 * s_ptb)
                lbg.append(-self.INF)
                ubg.append(self.P_max_tb_b)

            if self.enforce_stable:
                coef_i = well_data["coeff_stability"]
                u1_j = U[j][0]
                u2_j = U[j][1]
                b_i = polyval_casadi(u1_j, coef_i)

                # Stability feasible condition:
                # b_i(u1_j) - u2_j <= 0
                s_stab = ca.MX.sym(f"s_stab_{j}")
                slack_vars.append(s_stab)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(b_i - u2_j - s_stab)
                lbg.append(-self.INF)
                ubg.append(0.0)

        # ---------------------
        # 2.2) Platform constraints with slacks
        # ---------------------
        if not self.unconstrained_platform:
            if self.G_available is not None:
                s_ginj = ca.MX.sym("s_platform_G_available")
                slack_vars.append(s_ginj)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(total_w_g_inj - s_ginj)
                lbg.append(-self.INF)
                ubg.append(self.G_available)

            if self.G_max_export is not None:
                s_gexp = ca.MX.sym("s_platform_G_max_export")
                slack_vars.append(s_gexp)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(total_w_g_res - s_gexp)
                lbg.append(-self.INF)
                ubg.append(self.G_max_export)

            if self.W_max is not None:
                s_w = ca.MX.sym("s_platform_W_max")
                slack_vars.append(s_w)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(total_w_w - s_w)
                lbg.append(-self.INF)
                ubg.append(self.W_max)

            if self.L_max is not None:
                s_l = ca.MX.sym("s_platform_L_max")
                slack_vars.append(s_l)
                slack_lbx.append(0.0)
                slack_ubx.append(self.INF)
                slack_x0.append(0.0)

                g_list.append(total_w_l_out - s_l)
                lbg.append(-self.INF)
                ubg.append(self.L_max)

        # ---------------------
        # 2.3) Refinement oil-retention constraint with slack
        # ---------------------
        if self.refinement:
            s_oil = ca.MX.sym("s_refinement_oil_allowance")
            slack_vars.append(s_oil)
            slack_lbx.append(0.0)
            slack_ubx.append(self.INF)
            slack_x0.append(0.0)

            # total_w_o >= total_wo - allowance
            g_list.append(total_w_o + s_oil)
            lbg.append(self.total_wo - self.oil_allowance)
            ubg.append(self.INF)

        # ---------------------
        # 2.4) Trust-region constraint
        # ---------------------
        x_k_np = np.array(self.u_k, dtype=float).reshape((-1, 1))
        x_k = ca.DM(x_k_np)

        dx = x - x_k

        g_list.append(ca.dot(dx, dx))
        lbg.append(-self.INF)
        ubg.append(float(self.Delta ** 2))

        # ---------------------
        # 3) Full NLP variable vector
        # ---------------------
        if slack_vars:
            s = ca.vertcat(*slack_vars)
            x_full = ca.vertcat(x, s)
        else:
            s = ca.MX()
            x_full = x

        # ---------------------
        # 4) Objective: weighted slack norm
        # ---------------------
        if slack_vars:
            obj = ca.dot(s, s)
        else:
            obj = 0.0

        # ---------------------
        # 5) Bounds and initial guess
        # ---------------------
        lbx_u = []
        ubx_u = []

        for _ in range(self.N):
            lbx_u += list(self.u_lb)
            ubx_u += list(self.u_ub)

        lbx_full = ca.DM(lbx_u + slack_lbx).reshape((-1, 1))
        ubx_full = ca.DM(ubx_u + slack_ubx).reshape((-1, 1))

        x0_full = ca.DM(
            list(np.array(self.u_k, dtype=float).reshape((-1,))) + slack_x0
        ).reshape((-1, 1))

        g = ca.vertcat(*g_list) if g_list else ca.MX()

        # ---------------------
        # 6) Solve NLP
        # ---------------------
        nlp = {
            "x": x_full,
            "f": obj,
            "g": g,
        }

        solver = ca.nlpsol(
            "phase_zero_solver",
            "ipopt",
            nlp,
            self.ipopt_opts,
        )

        sol = solver(
            x0=x0_full,
            lbx=lbx_full,
            ubx=ubx_full,
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
        )

        # ---------------------
        # 7) Post-process solution
        # ---------------------
        x_full_star = np.array(sol["x"], dtype=float).reshape((-1,))
        x_star = x_full_star[: self.N * self.nu]

        if slack_vars:
            s_star = x_full_star[self.N * self.nu:]
        else:
            s_star = np.array([])

        x_k_flat = np.array(self.u_k, dtype=float).reshape((-1,))
        dx_star = x_star - x_k_flat

        tr_norm_sq = float(dx_star @ dx_star)
        tr_radius_sq = float(self.Delta ** 2)
        tr_violation = max(0.0, tr_norm_sq - tr_radius_sq)
        tr_feasible = tr_violation <= 1e-6
        tr_active = tr_feasible and tr_norm_sq >= 0.999 * tr_radius_sq

        g_star = np.array(sol["g"], dtype=float).reshape((-1,))

        # ---------------------
        # 8) Return
        # ---------------------
        return {
            "success": bool(solver.stats().get("success", False)),
            "stats": solver.stats(),

            "x_star": x_star,
            "x_full_star": x_full_star,
            "s_star": s_star,

            "objective_value": float(sol["f"]),

            "tr_norm_sq": tr_norm_sq,
            "tr_radius_sq": tr_radius_sq,
            "tr_violation": tr_violation,
            "tr_feasible": tr_feasible,
            "tr_active": tr_active,

            "g_star": g_star,
        }