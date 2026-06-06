
# def flatten_sweep_results_to_batch_full(results: dict,
#                                         only_success: bool = True,
#                                         y_names=None,
#                                         z_names=None):
#     """
#     Like flatten_sweep_results_to_batch, but also returns z targets from OUT.
#
#     Assumes:
#       - first 3 Z_NAMES are states y = [y1,y2,y3]
#       - z targets are selected explicitly by name
#     """
#     import numpy as np
#     import torch
#
#     if y_names is None:
#         y_names = ["m_G_an", "m_G_t", "m_o_t"]
#
#     if z_names is None:
#         z_names = ["P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res"]
#
#
#     u1_grid = np.asarray(results["u1_grid"], dtype=float)
#     u2_grid = np.asarray(results["u2_grid"], dtype=float)
#
#     # Z_NAMES = list(results["Z_NAMES"])
#     OUT = results["OUT"]
#
#     # SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
#     # RES_DX = np.asarray(results["RES_DX"], dtype=float)
#
#     Nu1 = len(u1_grid)
#     Nu2 = len(u2_grid)
#
#     # u grid
#     U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
#
#     u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)
#
#     # Flatten state inputs y
#     # ------------------------------------------------------------
#     y_flat = np.stack(
#         [
#             np.asarray(OUT[name], dtype=float).reshape(-1)
#             for name in y_names
#         ],
#         axis=1,
#     )
#
#     # -------------------------------
#
#
#     # y (first 3)
#     y_cols = []
#     for name in Z_NAMES[:3]:
#         arr = np.asarray(OUT[name], dtype=float)
#         y_cols.append(arr.reshape(-1))
#     y_flat = np.stack(y_cols, axis=1)
#
#     # z targets
#     z_names = ["P_bh_bar", "P_tb_b_bar","w_G_inj","w_res"]
#     z_cols = []
#     for name in z_names:
#         arr = np.asarray(OUT[name], dtype=float)
#         z_cols.append(arr.reshape(-1))
#     z_flat = np.stack(z_cols, axis=1)
#
#     success_flat = SUCCESS.reshape(-1)
#     res_dx_flat = RES_DX.reshape(-1)
#
#     finite_y = np.all(np.isfinite(y_flat), axis=1)
#     finite_z = np.all(np.isfinite(z_flat), axis=1)
#     finite_all = finite_y & finite_z
#
#     mask = (success_flat & finite_all) if only_success else finite_all
#
#     u_np = u_flat[mask]
#     y_np = y_flat[mask]
#     z_np = z_flat[mask]
#     res_dx_np = res_dx_flat[mask]
#
#     # Torch tensors
#     u_t = torch.tensor(u_np, dtype=torch.float32)
#     y_t = torch.tensor(y_np, dtype=torch.float32)
#     z_t = torch.tensor(z_np, dtype=torch.float32)
#     res_dx_t = torch.tensor(res_dx_np, dtype=torch.float32)
#
#     return {
#         "Z_NAMES": Z_NAMES,
#         "z_names": z_names,
#         "u_np": u_np,
#         "y_np": y_np,
#         "z_np": z_np,
#         "res_dx_np": res_dx_np,
#         "u_t": u_t,
#         "y_t": y_t,
#         "z_t": z_t,
#         "res_dx_t": res_dx_t,
#         "Nu1": Nu1,
#         "Nu2": Nu2,
#         "mask_np": mask,
#     }


# def prepare_data(self):
#     """
#     Prepare AlgNN data using ground-truth sweeps.
#
#     Dataset:
#         input  -> (y,u)
#         target -> (P_bh_bar, P_tb_b_bar)
#
#     States y are the first 3 entries in Z_NAMES.
#     Targets are selected by name.
#     """
#
#     self.l.info("Preparing AlgNN dataset from ground-truth sweeps...")
#
#     # ------------------------------------------------
#     # 1. Build sweep grids
#     # ------------------------------------------------
#     # Nu1_train = int(np.sqrt(self.N_train))
#     # Nu2_train = int(np.sqrt(self.N_train))
#     #
#     # Nu1_val = int(np.sqrt(self.N_val))
#     # Nu2_val = int(np.sqrt(self.N_val))
#
#     # u1_grid_train = np.linspace(self.u_min_train[0], self.u_max_train[0], Nu1_train)
#     # u2_grid_train = np.linspace(self.u_min_train[1], self.u_max_train[1], Nu2_train)
#     #
#     # u1_grid_val = np.linspace(self.u_min_train[0], self.u_max_train[0], Nu1_val)
#     # u2_grid_val = np.linspace(self.u_min_train[1], self.u_max_train[1], Nu2_val)
#
#     self.model_sur = make_model("surrogate",
#                                 BSW=self.BSW,
#                                 GOR=self.GOR,
#                                 PI=self.PI,
#                                 K_gs=self.K_gs,
#                                 K_inj=self.K_inj,
#                                 K_pr=self.K_pr)
#
#     self.y_guess_surr = np.array(self.y_guess_sur, dtype=float)
#     # ------------------------------------------------
#     # 2. Run sweeps
#     # ------------------------------------------------
#     self.l.info("Running TRAIN sweep...")
#     results_train = run_sweep(
#         self.model_sur,
#         U1_MIN=self.u1_min,
#         U2_MIN=self.u2_min,
#         U_SIM_SIZE=self.N_train,
#         y_guess_init=self.y_guess_surr)
#
#
#     self.l.info("Running VAL sweep...")
#     results_val = run_sweep(
#         self.model_sur,
#         U1_MIN=self.u1_min,
#         U2_MIN=self.u2_min,
#         U_SIM_SIZE=self.N_val,
#         y_guess_init=self.y_guess_surr)
#
#     # ------------------------------------------------
#     # 3. Flatten sweep results
#     # ------------------------------------------------
#     batch_train = flatten_sweep_results_to_batch_full(results_train, only_success=True)
#     batch_val = flatten_sweep_results_to_batch_full(results_val, only_success=True)
#
#     Z_NAMES = list(batch_train["Z_NAMES"])
#
#     # Inputs
#     y_train = batch_train["y_t"].to(self.device)
#     u_train = batch_train["u_t"].to(self.device)
#
#     y_val = batch_val["y_t"].to(self.device)
#     u_val = batch_val["u_t"].to(self.device)
#
#     # Targets
#     z_train = batch_train["z_t"].to(self.device)
#     z_val = batch_val["z_t"].to(self.device)
#
#     # Metadata
#     state_names = list(batch_train["Z_NAMES"][:3])
#     target_names = list(batch_train["z_names"])
#
#     # # ------------------------------------------------
#     # # 4. Extract targets by name
#     # # ------------------------------------------------
#     # target_names = ["P_bh_bar", "P_tb_b_bar"]
#     #
#     # def extract_targets(results, mask_np):
#     #     z_cols = []
#     #     for name in target_names:
#     #         arr = np.asarray(results["OUT"][name], dtype=float)
#     #         z_cols.append(arr.reshape(-1))
#     #     z_flat = np.stack(z_cols, axis=1)
#     #     z_np = z_flat[mask_np]
#     #     return torch.tensor(z_np, dtype=torch.float32)
#     #
#     # z_train = extract_targets(results_train, batch_train["mask_np"]).to(self.device)
#     # z_val = extract_targets(results_val, batch_val["mask_np"]).to(self.device)
#
#     # ------------------------------------------------
#     # 5. Store datasets
#     # ------------------------------------------------
#     self.train_data = (y_train, u_train, z_train)
#     self.val_data = (y_val, u_val, z_val)
#
#     self.l.info(
#         f"Data preparation complete.\n"
#         f"State inputs:  {state_names}\n"
#         f"Target outputs: {target_names}\n"
#         f"Train shapes: y={tuple(y_train.shape)}, u={tuple(u_train.shape)}, z={tuple(z_train.shape)}\n"
#         f"Val shapes:   y={tuple(y_val.shape)}, u={tuple(u_val.shape)}, z={tuple(z_val.shape)}"
#     )


# def _log_epoch_info(self, train_losses: dict, val_losses: dict):
#     """Implements the abstract logging method."""
#     train_loss = train_losses['total']
#     val_loss = val_losses['total']
#     val_pbh_loss = val_losses.get('mse_P_bh', 0.0)  # .get for safety
#
#     self.l.info(
#         f"Epoch {self._e} | "
#         f"Train Loss: {train_loss:.8f} | "
#         f"Val Loss: {val_loss:.8f} | "
#         f"(Val P_bh MSE: {val_pbh_loss:.6f})"
#     )
