
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

#
#
# def extract_17_point_values(poly_dataset, variable_name):
#     """
#     Extract the 17 selected values for the requested variable.
#
#     The dataset contains:
#         Y_pinn for states:
#             ["m_G_an", "m_G_t", "m_o_t"]
#
#         Y_poly for algebraic polynomial outputs:
#             ["P_bh_bar", "P_tb_b_bar", "w_G_inj"]
#     """
#
#     if variable_name in poly_dataset["pinn_output_names"]:
#         names = poly_dataset["pinn_output_names"]
#         Y = np.asarray(poly_dataset["Y_pinn"], dtype=float)
#
#     elif variable_name in poly_dataset["poly_output_names"]:
#         names = poly_dataset["poly_output_names"]
#         Y = np.asarray(poly_dataset["Y_poly"], dtype=float)
#
#     else:
#         raise ValueError(
#             f"Variable {variable_name} not found. Available variables are:\n"
#             f"PINN: {poly_dataset['pinn_output_names']}\n"
#             f"POLY: {poly_dataset['poly_output_names']}"
#         )
#
#     k = names.index(variable_name)
#
#     U_17 = np.asarray(poly_dataset["U"], dtype=float)
#     Y_17 = Y[:, k]
#
#     return U_17, Y_17
#
#
# # ============================================================
# # Main diagnostic plot
# # ============================================================
#
# def plot_sweep_surface_with_17_points_and_random_samples(
#     well_name,
#     variable_name,
#     elev=15,
#     azim=20,
#     n_random=20,
#     seed=123,
# ):
#     folder = get_well_folder(well_name)
#
#     sweep_path = folder / "sweep_results.pkl"
#     dataset_path = folder / "poly_dataset.pkl"
#
#     print(f"Loading sweep from:   {sweep_path}")
#     print(f"Loading dataset from: {dataset_path}")
#
#     sweep = load_pickle(sweep_path)
#     poly_dataset = load_pickle(dataset_path)
#
#     U1 = np.asarray(sweep["U1"], dtype=float)
#     U2 = np.asarray(sweep["U2"], dtype=float)
#
#     u1_min, u1_max = np.nanmin(U1), np.nanmax(U1)
#     u2_min, u2_max = np.nanmin(U2), np.nanmax(U2)
#
#     Y_sweep = np.asarray(sweep["OUT"][variable_name], dtype=float)
#
#     U_17, Y_17 = extract_17_point_values(poly_dataset, variable_name)
#
#     # =====================================================
#     # Random samples in [u1_min, 1.0] x [u2_min, 1.0]
#     # =====================================================
#     rng = np.random.default_rng(seed)
#
#     U_random = np.column_stack([
#         rng.uniform(u1_min, 1.0, size=n_random),
#         rng.uniform(u2_min, 1.0, size=n_random),
#     ])
#
#     # Put random samples on the bottom plane
#     z_floor = np.nanmin(Y_sweep) - 5.0
#     Z_random_floor = np.full(n_random, z_floor)
#
#     # =====================================================
#     # Figure
#     # =====================================================
#     fig = plt.figure(figsize=(7.2, 4.0))
#
#     ax_left = fig.add_subplot(1, 2, 1, projection="3d")
#     ax_right = fig.add_subplot(1, 2, 2, projection="3d")
#
#     axes = [ax_left, ax_right]
#
#     # Common z limits so both panels are visually comparable
#     z_min = z_floor
#     z_max = np.nanmax(Y_sweep)
#
#     for ax in axes:
#         ax.plot_surface(
#             U1,
#             U2,
#             Y_sweep,
#             linewidth=0,
#             antialiased=True,
#             alpha=0.70,
#             cmap="viridis",
#         )
#
#         ax.set_xlabel("$u_1$")
#         ax.set_ylabel("$u_2$")
#         ax.set_zlabel(variable_name)
#
#         ax.set_xlim(u1_min, u1_max)
#         ax.set_ylim(u2_min, u2_max)
#         ax.set_zlim(z_min, z_max)
#
#         ax.set_box_aspect((1.0, 1.0, 1.0))
#         ax.view_init(elev=elev, azim=azim)
#
#     # =====================================================
#     # Left: original 17 selected points
#     # =====================================================
#     ax_left.scatter(
#         U_17[:, 0],
#         U_17[:, 1],
#         Y_17,
#         marker="o",
#         s=50,
#         color="red",
#         edgecolor="black",
#         depthshade=True,
#     )
#     red_scatter=ax_right.scatter(
#         U_17[:, 0],
#         U_17[:, 1],
#         Y_17,
#         marker="o",
#         s=50,
#         color="red",
#         edgecolor="black",
#         depthshade=True,
#     )
#
#
#
#     # =====================================================
#     # Right: 50 random points on bottom plane
#     # =====================================================
#     blue_scatter=ax_right.scatter(
#         U_random[:, 0],
#         U_random[:, 1],
#         Z_random_floor,
#         marker="x",
#         s=50,
#         color="blue",
#         linewidths=1.8,
#         depthshade=False,
#     )
#
#
#     ax_left.set_title(
#         f"Polynomial interpolation", fontsize=14,pad=-10
#     )
#     ax_right.set_title(
#         f"PINN-AlgNN Training", fontsize=14, pad=-10
#     )
#
#     fig.legend(
#         handles=[red_scatter, blue_scatter],
#         labels=["Simulated data points", "Collocation points"],
#         loc="lower center",
#         ncol=2,
#         frameon=False,
#         bbox_to_anchor=(0.5, -0.005),
#         fontsize=12,
#
#         handletextpad=0.05,   # distance between marker and text
#         columnspacing=1.5,   # distance between the two legend entries
#     )
#
#     fig.subplots_adjust(
#         left=0.06,
#         right=0.98,
#         bottom=0.11,
#         top=0.98,
#         wspace=0.12,
#         hspace=0.10,
#     )
#
#     fig.savefig(
#         "experiment_design.pdf",
#         format="pdf"
#     )
#
#     plt.show()
#
#     return {
#         "U1": U1,
#         "U2": U2,
#         "Y_sweep": Y_sweep,
#         "U_17": U_17,
#         "Y_17": Y_17,
#         "U_random": U_random,
#         "z_floor": z_floor,
#     }
#
# # if __name__ == "__main__":
# #     data = plot_sweep_surface_with_17_points(
# #         well_name=WELL_NAME,
# #         variable_name=VARIABLE_NAME,
# #     )
#
# if __name__ == "__main__":
#     data = plot_sweep_surface_with_17_points_and_random_samples(
#         well_name=WELL_NAME,
#         variable_name=VARIABLE_NAME,
#         n_random=40,
#         seed=123,
#     )


# if self._e % 1000 == 0 and z is not None:
#     with torch.no_grad():
#         abs_dx = torch.abs(dx)
#         worst_flat_idx = torch.argmax(abs_dx)
#         worst_row = worst_flat_idx // dx.shape[1]
#         worst_comp = worst_flat_idx % dx.shape[1]
#
#         self.l.info("========== WORST PHYSICS POINT ==========")
#         self.l.info(f"epoch: {self._e}")
#         self.l.info(f"worst row: {int(worst_row)}")
#         self.l.info(f"worst component: {int(worst_comp)}")
#         self.l.info(f"u worst: {u[worst_row]}")
#         self.l.info(f"y_hat worst: {y_hat[worst_row]}")
#         self.l.info(f"dx worst row: {dx[worst_row]}")
#         self.l.info(f"dx_scaled worst row: {dx_scaled[worst_row]}")
#         self.l.info("========================================")
#
#         self.l.info("========== ALGEBRAIC VALUES AT WORST POINT ==========")
#
#         z_worst = z[worst_row]
#
#         for name, value in zip(Z_DIAG_NAMES, z_worst):
#             self.l.info(f"{name:25s}: {float(value.item()): .6e}")
#         u_worst_np = u[worst_row].detach().cpu().numpy()
#         y_worst_np = y_hat[worst_row].detach().cpu().numpy()
#
#         self.debug_nearest_validation_state(u_worst_np, y_worst_np)
#
#         self.debug_compare_predicted_vs_validation_algebraics(
#             u[worst_row].detach().cpu().numpy(),
#             y_hat[worst_row].detach().cpu().numpy(),
#         )
#
#         self.l.info("=====================================================")
#
#