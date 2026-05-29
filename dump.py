
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


# # --------------------------------------------------
# # 5) CHECK interpolation at u_k_j
# # --------------------------------------------------
# z_corr_at_uk = np.array(F_corr_j(u=u_k_j)["z"], dtype=float).reshape((-1,))
#
# print("\n--- Interpolation check at u_k_j ---")
# print("z_corr_j(u_k_j) =", z_corr_at_uk)
# print("z_p_j           =", z_p_j)
# print("z_corr_j(u_k_j) - z_p_j =")
# print(z_corr_at_uk - z_p_j)
# print("||z_corr_j(u_k_j) - z_p_j||_2 =", np.linalg.norm(z_corr_at_uk - z_p_j))
#
# # --------------------------------------------------
# # 6) CHECK corrected Jacobian at u_k_j
# # --------------------------------------------------
# J_corr_sym = ca.jacobian(z_corr_j, u_j)
# F_corr_eval = ca.Function(
#     f"F_corr_eval_{well_name}",
#     [u_j],
#     [z_corr_j, J_corr_sym],
#     ["u"],
#     ["z", "J"]
# )
#
# corr_eval = F_corr_eval(u=u_k_j)
# z_corr_eval = np.array(corr_eval["z"], dtype=float).reshape((-1,))
# J_corr_eval = np.array(corr_eval["J"], dtype=float)
#
# print("\n--- Jacobian consistency check at u_k_j ---")
# print("J_corr_j(u_k_j) =")
# print(J_corr_eval)
# print("J_p_j =")
# print(J_p_j)
# print("J_corr_j(u_k_j) - J_p_j =")
# print(J_corr_eval - J_p_j)
# print("||J_corr_j(u_k_j) - J_p_j||_F =", np.linalg.norm(J_corr_eval - J_p_j))


###################
#ANNULUS CHOKE

# import numpy as np
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import plotly.graph_objects as go
# #
# #
# # def ksi(P_down, P_up, gamma=1.3, eps=1e-8):
# #     P_up_safe = np.maximum(P_up, eps)
# #     r = P_down / P_up_safe
# #     val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
# #
# #     return val
# #
# # K_gs = 9.98e-5  # is the gas lift choke constant
# # K_gs_new=K_gs
# # T_an = 348  # K is the annulus temperature
# #
# # u2=np.linspace(0.1,1.001,60)
# # P_at=np.linspace(90e5,145e5,60)
# #
# # U2, P_AT = np.meshgrid(u2, P_at)
# #
# # R = 8.314  # J/(K*mol) is the universal gas constant
# # M_G = 0.0167  # (kg/mol) is the gas molecular weight
# #
# # P_gs = 140e5  # 140bar is the gas source pressure
# # rho_G_in = P_gs * M_G / (R * T_an)  # 2.4
# #
# # scale = P_gs * np.sqrt(M_G / (R * T_an))
# #
# # w_G_in_atual=K_gs*U2*np.sqrt(np.maximum(rho_G_in*(P_gs-P_AT),0.0))
# # w_G_in_new=K_gs_new*U2*scale*np.sqrt(np.maximum(ksi(P_AT,P_gs),0.0))
# #
# # # Combined 3D plot
# # # =========================
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# #
# #
# # # Old model (blue)
# # surf1 = ax.plot_surface(
# #     U2, P_AT / 1e5, w_G_in_atual,
# #     color='Blue',
# #     edgecolor='none'
# # )
# #
# # # New model (red)
# # surf2 = ax.plot_surface(
# #     U2, P_AT / 1e5, w_G_in_new,
# #     color='Red',
# #     edgecolor='none'
# # )
# #
# # ax.set_xlabel('u2')
# # ax.set_ylabel('P_at [bar]')
# # ax.set_zlabel('w_G_in')
# # ax.set_title('Comparison: Old vs New Valve Law')
# #
# # plt.tight_layout()
# # plt.show()
#
# import numpy as np
# import plotly.graph_objects as go
#
# k_pos=2000
# eps = 1e-12
#
# def softplus_stable(x):
#     ax = np.sqrt(x * x + eps)
#     return 0.5 * (x + ax) + np.log(1 + np.exp(-ax))
#
#
# def smooth_pos_scaled(dp_pa):
#     x = (k_pos * dp_pa)
#     return (1 / k_pos) * (softplus_stable(x))
#
#
# def smooth_max_scaled(z, zmin):
#     # smooth approximation of max(z,zmin))
#     return zmin + smooth_pos_scaled(z - zmin)
#
# def ksi(P_down, P_up, gamma=1.3, eps=1e-8):
#     P_up_safe = np.maximum(P_up, eps)
#     r = P_down / P_up_safe
#     val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
#     return smooth_pos_scaled(val)
#
# def ksi_atual(P_down, P_up, gamma=1.3, eps=1e-8):
#     P_up_safe = np.maximum(P_up, eps)
#     r = P_down / P_up_safe
#     val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
#     return val
#
# K_gs = 9.98e-5
# K_gs_new = K_gs
# T_an = 348
#
# u2 = np.linspace(0.1, 1.001, 60)
# P_at = np.linspace(90e5, 145e5, 60)
#
# U2, P_AT = np.meshgrid(u2, P_at)
#
# R = 8.314
# M_G = 0.0167
# P_gs = 140e5
#
# rho_G_in = P_gs * M_G / (R * T_an)
# scale = P_gs * np.sqrt(M_G / (R * T_an))
#
# # w_G_in_atual = K_gs * U2 * np.sqrt(rho_G_in*smooth_pos_scaled(P_gs - P_AT))
# w_G_in_atual = K_gs_new * U2 * scale * np.sqrt(np.maximum(ksi_atual(P_AT, P_gs),0))
# w_G_in_new = K_gs_new * U2 * scale * np.sqrt(ksi(P_AT, P_gs))
#
# fig = go.Figure()
#
# fig.add_trace(go.Surface(
#     x=U2,
#     y=P_AT / 1e5,
#     z=w_G_in_atual,
#     colorscale='Blues',
#     opacity=0.75,
#     name='Old model',
#     showscale=False
# ))
#
# fig.add_trace(go.Surface(
#     x=U2,
#     y=P_AT / 1e5,
#     z=w_G_in_new,
#     colorscale='Reds',
#     opacity=0.75,
#     name='New model',
#     showscale=False
# ))
#
# fig.update_layout(
#     title='Comparison: Old vs New Valve Law',
#     scene=dict(
#         xaxis_title='u2',
#         yaxis_title='P_at [bar]',
#         zaxis_title='w_G_in'
#     ),
#     width=950,
#     height=750
# )
#
# fig.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# tiny=1e-12
# eps=1e-12
# k_pos=50
# dp_pa=1e5
#
#
# def softplus_stable(x):
#     # softplus(x) = max(x,0) + log(1+exp(-|x|)), stable for large x
#     ax = np.sqrt(x * x + tiny)
#     return 0.5 * (x + ax) + np.log(1 + np.exp(-ax))
#
#
# def smooth_pos_scaled(dp_pa, scale=1):
#     x = (k_pos * dp_pa) / scale
#     # softplus(z) = (1/k) log(1+exp(k z))
#     return (scale / k_pos) * (softplus_stable(x))
#
# K_inj=1.40e-4
# rho_G_in=80
# # dp range: 0 to 100000 Pa
# dp = np.linspace(-0.5, 0.5, 1000)
#
# smooth=smooth_pos_scaled(dp,scale=1)
# nonsmooth= np.maximum(dp, 0)
#
# # Plot both on the same figure
# plt.figure()
# plt.plot(dp, smooth, label="smooth_pos_scaled")
# plt.plot(dp, nonsmooth, label="max(dp, 0)")
# plt.xlabel("dp (Pa)")
# plt.ylabel("w_G_in")
# plt.title("Comparison: Smooth vs max(dp, 0)")
# plt.legend()
# plt.show()



#PRESSURES ANNULUS

# from casadi import *
# import numpy as np
# import casadi as ca
#
#
# k_pos = 20
# eps = 1e-12
#
# # # Well properties ###
# BSW = 0
# GOR = 0 # is the gas oil ratio
# PI = 3.00e-6  # is the productivity index in kg/(s.Pa)
#
# # Geometry and temperature of the wellbore
# ### Annulus ###
# V_an = 64.34  # m^3 is the annulus volume
# L_an = 2048  # m is the length of the annulus
# T_an = 348  # K is the annulus temperature
#
# ### Tubing bottom ###
# D_bh=0.2
# L_bh=75
# S_bh=(np.pi*D_bh**2)/4
# V_bh = S_bh * L_bh
# T_bh = 371.5
#
# ### Tubing ###
# L_tb = 1973
# D_tb = 0.134
# S_tb = (np.pi*D_tb**2)/4
# V_tb = S_tb * L_tb
# T_tb = 369.4  # K is the tubing temperature
#
# V_tb=V_tb-V_bh
#
# # Constants (general)
# R = 8.314  # J/(K*mol) is the universal gas constant
# g = 9.81  # m/s^2 is the gravity
# mu_o = 3.64e-3  # Pa.s is the viscosity
# mu_w = 1.00e-3
# rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
# rho_w = 1000
# rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
# mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
# M_G = 0.0167  # (kg/mol) is the gas molecular weight
#
# # Pressures
# P_gs = 140e5  # 140bar is the gas source pressure
# P_res = 160e5  # 160bar, the constant reservoir pressure
# P_0 = 20e5  # pressure downstream of choke
#
# # Chokes
# K_gs = 9.98e-5  # is the gas lift choke constant
# # K_gs=1e-4
# K_inj = 1.40e-4  # is the injection valve choke constant
# K_pr = 2.90e-3  # is the production choke constant
# K0_int=2.50
#
# # Friction
# epsilon_tubing = 1e-3
#
# # ---------- unpack ----------
# # m_G_an=y[0]
# # m_G_t=y[1]
# # m_o_t=y[2]
# # m_w_t=y[3]
# # m_G_b=y[4]
# # m_o_b=y[5]
# # m_w_b =y[6]
# # u1=u[0]
# # u2=u[1]
#
# # Algebraic variables
# #
# # P_bh_g=z[0] #Pressure bottomhole, closed at g6
# # P_tb_b_g=z[1] # Pressure at the bottom of the tubing, closed at g7
# # w_res_g=z[2] # Reservoir flow in kg/s
# # w_up_g=z[3] # Flow into the tubing in kg/s
#
#
# # -----------------------
# # PART 1 - RESERVOIR INFLOW
# # -----------------------
# m_G_an = np.linspace(3200.0, 4800.0, 300)
# # w_L_res=w_res_g/(1+GOR) # 1.1
# # w_o_res=(1-BSW)*w_L_res # 1.2
# # w_w_res=BSW*w_L_res  #1.3
# # w_G_res=GOR*w_L_res #1.4
#
#
# P_an_t_old=R*T_an*m_G_an/(M_G*V_an) # 2.1
# P_an_b_old=P_an_t_old+(m_G_an*g*L_an/V_an) # 2.2
#
# P_an_t_new=(m_G_an*g/(V_an/L_an))*(
#     np.exp(-g*M_G*L_an/(R*T_an)) /(1-np.exp(-g*M_G/(R*T_an)*L_an)))
#
# P_an_b_new=(m_G_an*g/(V_an/L_an))*(
#         1/(1-np.exp(-g*M_G/(R*T_an)*L_an)))
#
# import numpy as np
# import matplotlib.pyplot as plt
# # -----------------------------
#
# # -----------------------------
# # Convert to bar
# # -----------------------------
# P_an_t_old_bar = P_an_t_old / 1e5
# P_an_b_old_bar = P_an_b_old / 1e5
# P_an_t_new_bar = P_an_t_new / 1e5
# P_an_b_new_bar = P_an_b_new / 1e5
#
# # -----------------------------
# # Plot 1: all 4 curves together
# # -----------------------------
# plt.figure(figsize=(9, 6))
# plt.plot(m_G_an, P_an_t_old_bar, label='Old $P_{an,t}$', linewidth=2)
# plt.plot(m_G_an, P_an_b_old_bar, label='Old $P_{an,b}$', linewidth=2)
# plt.plot(m_G_an, P_an_t_new_bar, label='New $P_{an,t}$', linewidth=2, linestyle='--')
# plt.plot(m_G_an, P_an_b_new_bar, label='New $P_{an,b}$', linewidth=2, linestyle='--')
#
# plt.xlabel(r'$m_{G,an}$ [kg]')
# plt.ylabel('Pressure [bar]')
# plt.title('Annulus pressure comparison vs gas mass')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # -----------------------------
# # Plot 2: difference old vs new
# # -----------------------------
# plt.figure(figsize=(9, 6))
# plt.plot(m_G_an, P_an_t_new_bar, label='new top', linewidth=2)
# plt.plot(m_G_an,P_an_t_old_bar, label='old top', linewidth=2)
# plt.plot(m_G_an, P_an_b_new_bar, label='new bottom', linewidth=2)
# plt.plot(m_G_an, P_an_b_old_bar, label='old bottom', linewidth=2)
# plt.xlabel(r'$m_{G,an}$ [kg]')
# plt.ylabel('Pressure difference [bar]')
# plt.title('Difference between new and old annulus pressure models')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



#PRESSURES TUBING


from casadi import *
# import numpy as np
# import casadi as ca
#
# def make_glc_well_rigorous(BSW,GOR,PI):
#
#     def glc_well(y, z, u):
#
#         def softplus_stable(x):
#             ax = sqrt(x * x + eps)
#             return 0.5 * (x + ax) + log(1 + exp(-ax))
#
#         def smooth_pos_scaled(dp_pa):
#             x = (k_pos * dp_pa)
#             return (1 / k_pos) * (softplus_stable(x))
#
#         def smooth_max_scaled(z, zmin):
#             # smooth approximation of max(z,zmin))
#             return zmin + smooth_pos_scaled(z - zmin)
#
#         k_pos = 20
#         eps = 1e-12
#
#         # # Well properties ###
#         # BSW = 0
#         # GOR = 0 # is the gas oil ratio
#         # PI = 3.00e-6  # is the productivity index in kg/(s.Pa)
#
#         # Geometry and temperature of the wellbore
#         ### Annulus ###
#         V_an = 64.34  # m^3 is the annulus volume
#         L_an = 2048  # m is the length of the annulus
#         T_an = 348  # K is the annulus temperature
#
#         ### Tubing bottom ###
#         D_bh=0.2
#         L_bh=75
#         S_bh=(np.pi*D_bh**2)/4
#         V_bh = S_bh * L_bh
#         T_bh = 371.5
#
#         ### Tubing ###
#         L_tb = 1973
#         D_tb = 0.134
#         S_tb = (np.pi*D_tb**2)/4
#         V_tb = S_tb * L_tb
#         T_tb = 369.4  # K is the tubing temperature
#
#         V_tb=V_tb-V_bh
#
#         # Constants (general)
#         R = 8.314  # J/(K*mol) is the universal gas constant
#         g = 9.81  # m/s^2 is the gravity
#         mu_o = 3.64e-3  # Pa.s is the viscosity
#         mu_w = 1.00e-3
#         rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
#         rho_w = 1000
#         rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
#         mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
#         M_G = 0.0167  # (kg/mol) is the gas molecular weight
#
#         # Pressures
#         P_gs = 140e5  # 140bar is the gas source pressure
#         P_res = 160e5  # 160bar, the constant reservoir pressure
#         P_0 = 20e5  # pressure downstream of choke
#
#         # Chokes
#         K_gs = 9.98e-5  # is the gas lift choke constant
#         # K_gs=1e-4
#         K_inj = 1.40e-4  # is the injection valve choke constant
#         K_pr = 2.90e-3  # is the production choke constant
#         K0_int=2.50
#
#         # Friction
#         epsilon_tubing = 1e-3
#
#         # ---------- unpack ----------
#         m_G_an=y[0]
#         m_G_t=y[1]
#         m_o_t=y[2]
#         m_w_t=y[3]
#         m_G_b=y[4]
#         m_o_b=y[5]
#         m_w_b =y[6]
#         u1=u[0]
#         u2=u[1]
#
#         # Algebraic variables
#
#         P_bh_g=z[0] #Pressure bottomhole, closed at g6
#         P_tb_b_g=z[1] # Pressure at the bottom of the tubing, closed at g7
#         w_res_g=z[2] # Reservoir flow in kg/s
#         w_up_g=z[3] # Flow into the tubing in kg/s
#
#
#         # -----------------------
#         # PART 1 - RESERVOIR INFLOW
#         # -----------------------
#
#         w_L_res=w_res_g/(1+GOR) # 1.1
#         w_o_res=(1-BSW)*w_L_res # 1.2
#         w_w_res=BSW*w_L_res  #1.3
#         w_G_res=GOR*w_L_res #1.4
#
#         # -----------------------
#         # PART 2 - ANNULUS (from state)
#         # -----------------------
#         P_an_t=R*T_an*m_G_an/(M_G*V_an) # 2.1
#         P_an_b=P_an_t+(m_G_an*g*L_an/V_an) # 2.2
#
#         rho_G_an_b=P_an_b*M_G/(R*T_an) # 2.3
#         rho_G_in=P_gs*M_G/(R*T_an) # 2.4
#
#         dP_gs_an = P_gs - P_an_t # 2.5
#         w_G_in = K_gs * u2 * sqrt(rho_G_in * smooth_pos_scaled(dP_gs_an) + eps) # 2.6
#
#         # This makes our equation 2.7
#         dP_an_tb = P_an_b - P_tb_b_g
#         w_G_inj = K_inj * sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb) + eps)
#
#         # -----------------------
#         # PART 3 - DENSITIES USING STATES
#         # -----------------------
#
#         V_G_tb=V_tb-(m_o_t/rho_o)-(m_w_t/rho_w)
#         V_G_tb_safe = smooth_max_scaled(V_G_tb, 1e-6)
#
#         rho_G_tb = m_G_t/V_G_tb_safe
#
#         V_G_bh=V_bh-(m_o_b/rho_o)-(m_w_b/rho_w)
#         V_G_bh_safe=smooth_max_scaled(V_G_bh, 1e-6)
#
#         rho_G_bh=m_G_b/V_G_bh_safe
#
#         P_tb_t = rho_G_tb * R * T_tb / M_G
#         P_bh_t= rho_G_bh * R * T_bh / M_G
#
#         # -----------------------
#         # PART 4 - Hold-ups from states
#         # -----------------------
#
#         V_w_tb_states = m_w_t / rho_w
#         V_o_tb_states = m_o_t / rho_o
#         V_L_tb_states=V_w_tb_states+V_o_tb_states
#
#         V_w_bh_states = m_w_b / rho_w
#         V_o_bh_states = m_o_b / rho_o
#         V_L_bh_states=V_w_bh_states+V_o_bh_states
#
#         alpha_L_tb=(V_w_tb_states+V_o_tb_states)/V_tb
#         alpha_G_tb=1.0-alpha_L_tb
#
#         alpha_L_bh=(V_w_bh_states+V_o_bh_states)/V_bh
#         alpha_G_bh=1.0-alpha_L_bh
#
#         V_L_bh_states_safe = smooth_max_scaled(V_L_bh_states, 1e-9)
#         V_L_tb_states_safe = smooth_max_scaled(V_L_tb_states, 1e-9)
#
#         # Liquid compositions in each CV (for splitting oil/water)
#         f_w_b = V_w_bh_states / V_L_bh_states_safe
#         f_w_t = V_w_tb_states / V_L_tb_states_safe
#
#         # Densities in each CV (use constant rho_L in spirit of article; you can swap to state-based if desired)
#         # Using constant rho_L keeps consistency with the article’s closure style.
#         rho_L_tb = rho_L
#         rho_L_bh = rho_L
#
#         # Mixture density in each CV (volume averaged)
#         rho_mix_tb = (m_o_t+m_w_t+m_G_t)/V_tb
#         rho_mix_bh = (m_o_b+m_w_b+m_G_b)/V_bh
#         rho_mix_tb_safe = smooth_max_scaled(rho_mix_tb, 1e-9)
#         rho_mix_bh_safe = smooth_max_scaled(rho_mix_bh, 1e-9)
#
#         # -----------------------
#         # PART 5 - Decomposition of w_up
#         # -----------------------
#
#         alpha_G_bh_mass = (alpha_G_bh * rho_G_bh) / rho_mix_bh_safe
#         alpha_G_bh_mass = ca.fmin(ca.fmax(alpha_G_bh_mass, 0.0), 1.0)  # keep bounded
#
#         w_G_up = alpha_G_bh_mass * w_up_g
#         w_L_up = (1.0 - alpha_G_bh_mass) * w_up_g
#
#         w_w_up = f_w_b * w_L_up
#         w_o_up = (1.0 - f_w_b) * w_L_up
#
#         # -----------------------
#         # PART 6 - Superficial velocities
#         # -----------------------
#
#         # Superficial velocities bottom
#         U_sg_b=w_G_res/(rho_G_bh*S_bh) # 4.1
#         U_sl_b=w_L_res/(rho_L_bh*S_bh) # 4.2
#
#         # Superficial velocities top
#         U_sg_t=(w_G_up+w_G_inj)/(rho_G_tb*S_tb) # 4.3
#         U_sl_t=(w_L_up)/(rho_L_tb*S_tb) # 4.4
#
#         U_avg_b=U_sg_b+U_sl_b # 4.5
#         U_avg_t=U_sg_t+U_sl_t # 4.6
#
#         # -----------------------
#         # PART 7 - Friction and gravity
#         # -----------------------
#
#         # In the top section
#         Re_tb = rho_mix_tb_safe*U_avg_t*D_tb/mu
#         Re_tb_safe=smooth_max_scaled(Re_tb,1.0)  # keep >0 for 6.9/Re
#
#         log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
#         log_arg_tb_safe = fmax(log_arg_tb, 1e-12)
#         lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2
#
#         F_t = (alpha_L_tb*lambda_tb*rho_mix_tb*U_avg_t**2*L_tb)/(2*D_tb)
#
#         dP_t=rho_mix_tb*g*L_tb+F_t
#         P_tb_b=P_tb_t+dP_t
#
#         g1=P_tb_b-P_tb_b_g # Here we close the loop for the tubing bottom pressure
#
#         # In the bottom section
#         Re_bh = rho_mix_bh_safe*U_avg_b*D_bh/mu
#         Re_bh_safe=smooth_max_scaled(Re_bh, 1.0)
#
#         log_arg_bh=(epsilon_tubing/(D_bh*3.7))**1.11+6.9/Re_bh_safe
#         log_arg_bh_safe=fmax(log_arg_bh, 1e-12)
#         lambda_bh=(1.0/(1.8*log10(log_arg_bh_safe)))**2
#
#         F_bh=(lambda_bh*rho_mix_bh*U_avg_b**2*L_bh)/(2.0*D_bh)
#
#         dP_bh_seg=rho_mix_bh*g*L_bh+F_bh
#         P_bh=P_tb_b+dP_bh_seg
#
#         g2= P_bh-P_bh_g
#
#         # -----------------------
#         # PART 8 - Interface minor loss
#         # -----------------------
#         A_b=S_bh
#         A_t=S_tb #smaller
#         beta=A_t/A_b #area ratio
#
#         # Simple diameter-based minor-loss coefficient (smooth-ish & safe):
#         # You can tune K0_int to make ΔP small.
#         # - For contraction: (1/beta - 1)^2
#         K_con = (1.0/beta - 1.0)**2
#         K_int = K0_int + K_con
#
#         # INTERFACE HERE!
#         dP_int = K_int*(w_up_g**2)/(2.0*rho_mix_bh*(A_t**2)+eps)
#
#         g3=(P_bh_t-P_tb_b)-dP_int
#
#         # -----------------------
#         # PART 9 — Outflow at top choke (article-style)
#         # -----------------------
#
#         rho_G_tb_b=P_tb_b*M_G/(R*T_tb)
#
#         denom_alpha_b=(w_L_up*rho_G_tb_b+(w_G_inj+w_G_up)*rho_L_tb)
#         denom_alpha_b_safe=smooth_max_scaled(denom_alpha_b, 1e-9)
#
#         alpha_L_tb_b=w_L_up*rho_G_tb_b/denom_alpha_b_safe
#         alpha_L_tb_t = 2 * alpha_L_tb - alpha_L_tb_b
#
#         # Equation 30 - Mixture density at the top
#         rho_mix_tb_t = alpha_L_tb_t * rho_L + (1 - alpha_L_tb_t) * rho_G_tb
#         rho_mix_tb_t_safe = smooth_max_scaled(rho_mix_tb_t, 1e-9)
#
#         dP_tb_choke = P_tb_t - P_0
#         rho_G_tb_b=P_tb_b*M_G/(R*T_tb)
#         w_out = K_pr * u1 * ca.sqrt(rho_mix_tb_t_safe * smooth_pos_scaled(dP_tb_choke) + eps)
#
#         denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb
#         denom_alpha_t_safe = fmax(denom_alpha_t, 1e-12)
#
#         # Equation 33 - Gas Mass Fraction at top of tubing
#         alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb) / denom_alpha_t_safe
#
#         w_G_out = alpha_G_tb_t * w_out
#
#         w_L_out = (1.0 - alpha_G_tb_t) * w_out
#
#         w_w_out = f_w_t * w_L_out
#         w_o_out = (1.0 - f_w_t) * w_L_out
#
#         # ============================================================
#         # PART 8 — ODEs (states evolve)
#         # ============================================================
#         # Annulus gas
#         dx_mG_an = w_G_in - w_G_inj
#
#         # Bottom CV
#         dx_mG_b  = w_G_res - w_G_up
#         dx_mo_b  = w_o_res - w_o_up
#         dx_mw_b  = w_w_res - w_w_up
#
#         # Top CV (injection enters top)
#         dx_mG_t  = w_G_up + w_G_inj - w_G_out
#         dx_mo_t  = w_o_up           - w_o_out
#         dx_mw_t  = w_w_up           - w_w_out
#
#         dx = ca.vertcat(dx_mG_an, dx_mG_t, dx_mo_t, dx_mw_t, dx_mG_b, dx_mo_b, dx_mw_b)
#
#         # ============================================================
#         # PART 9 — Algebraic residuals g(x,z,u)=0
#         # ============================================================
#
#         # (G1) PI closure: w_res = PI * max(P_res - P_bh, 0)
#         dP_res_bh = P_res - P_bh
#         w_res=PI*smooth_pos_scaled(dP_res_bh) # This is closure for w_res
#         g4=w_res_g-w_res
#
#
#         alg = vertcat(g1, g2,g3,g4)
#
#         P_bh_bar = P_bh / 1e5
#         P_tb_t_bar = P_tb_t / 1e5
#         P_tb_b_bar = P_tb_b / 1e5
#         P_an_t_bar=P_an_t / 1e5
#         P_an_b_bar=P_an_b / 1e5
#
#         P_hidro_tb_bar=(rho_mix_tb*g*L_tb)/1e5
#         P_hidro_bh_bar=(rho_mix_bh*g*L_bh)/1e5
#
#         dP_int_bar=dP_int / 1e5
#         dP_gs_an_bar = dP_gs_an / 1e5
#         dP_an_tb_bar = dP_an_tb / 1e5
#         dP_res_bh_bar = dP_res_bh / 1e5
#         dP_tb_choke_bar = dP_tb_choke / 1e5
#         F_t_bar = F_t / 1e5
#         F_bh_bar = F_bh / 1e5
#
#         out = ca.vertcat(
#
#             # =================================
#             # States (masses in kg)
#             # =================================
#             m_G_an,
#             m_G_t,
#             m_o_t,
#             m_w_t,
#             m_G_b,
#             m_o_b,
#             m_w_b,
#
#             # =================================
#             # Hold-ups / Phase volumes (m³)
#             # =================================
#             V_L_tb_states,
#             V_L_bh_states,
#
#             alpha_L_tb,
#             alpha_G_tb,
#             alpha_L_tb_t,
#             alpha_L_tb_b,
#
#             alpha_L_bh,
#             alpha_G_bh,
#
#             # =================================
#             # Pressures (bar)
#             P_an_t_bar,
#             P_an_b_bar,
#             P_bh_bar,
#             P_tb_t_bar,
#             P_tb_b_bar,
#             P_hidro_tb_bar,
#             P_hidro_bh_bar,
#
#             # Friction (bar)
#             F_t_bar,
#             F_bh_bar,
#
#             # =================================
#             # Delta Pressures (bar)
#             # =================================
#             dP_int_bar,
#             dP_gs_an_bar,
#             dP_an_tb_bar,
#             dP_res_bh_bar,
#             dP_tb_choke_bar,
#
#             Re_tb,
#             Re_bh,
#             U_avg_t,
#             U_avg_b,
#
#             # =================================
#             # Flows (kg/s)
#             # =================================
#             w_res,
#             w_L_res,
#             w_G_res,
#
#             w_out,
#             w_G_out,
#             w_L_out,
#             w_w_out,
#             w_o_out,
#
#             w_G_inj,
#             w_up_g,
#             rho_mix_tb,
#         )
#         return dx, alg, out
#
#     return glc_well
#
# Z_NAMES_RIG = [
#
#     # =================================
#     # States (kg)
#     # =================================
#     "m_G_an",
#     "m_G_t",
#     "m_o_t",
#     "m_w_t",
#     "m_G_b",
#     "m_o_b",
#     "m_w_b",
#
#     # =================================
#     # Hold-ups / Volumes (m³)
#     # =================================
#     "V_L_tb_states",
#     "V_L_bh_states",
#     # Volume fractions
#     "alpha_L_tb",
#     "alpha_G_tb",
#     "alpha_L_tb_t",
#     "alpha_L_tb_b",
#     "alpha_L_bh",
#     "alpha_G_bh",
#
#
#     # =================================
#     # Pressures (bar)
#     # =================================
#     "P_an_t_bar",
#     "P_an_bar",
#     "P_bh_bar",
#     "P_tb_t_bar",
#     "P_tb_b_bar",
#     "P_hidro_tb_bar",
#     "P_hidro_bh_bar",
#
#     # Friction (bar)
#     "F_t_bar",
#     "F_bh_bar",
#
#     # =================================
#     # Delta Pressures (bar)
#     # =================================
#     "dP_int_bar",
#     "dP_gs_an_bar",
#     "dP_an_tb_bar",
#     "dP_res_bh_bar",
#     "dP_tb_choke_bar",
#     "Re_tb",
#     "Re_bh",
#     "U_avg_mix_tb",
#     "U_avg_b",
#
#     # =================================
#     # Flows (kg/s)
#     # =================================
#     "w_res",
#     "w_L_res",
#     "w_G_res",
#
#     "w_out",
#     "w_G_out",
#     "w_L_out",
#     "w_w_out",
#     "w_o_out",
#
#     "w_G_inj",
#
#     "w_up",
#     "rho_avg_mix_tb"
# ]
#
#
#
# #
# # def build_well_model(i: int, name_prefix="well"):
# #     """
# #     Returns a dict with explicit metadata + a compiled CasADi function:
# #         F_all(y,z,u) -> (dx, g, out)
# #     """
# #     well_id = i + 1
# #     fname = f"glc_well_{well_id:02d}_rigorous_casadi"
# #     if fname not in globals():
# #         raise AttributeError(f"No function '{fname}' in this module.")
# #     well_func = globals()[fname]
# #
# #     nx = 7
# #     nu = 2
# #     nz = 8   # your algebraic vector length, e.g. [P_tb_b, P_bh, w_res]
# #
# #     y = ca.MX.sym(f"y_{name_prefix}_{well_id}", nx)
# #     z = ca.MX.sym(f"z_{name_prefix}_{well_id}", nz)
# #     u = ca.MX.sym(f"u_{name_prefix}_{well_id}", nu)
# #
# #     dx, g, out = well_func(y, z, u)
# #
# #     F_all = ca.Function(
# #         f"F_all_{name_prefix}_{well_id}",
# #         [y, z, u],
# #         [dx, g, out],
# #         ["y", "z", "u"],
# #         ["dx", "g", "out"]
# #     )
# #
# #     return {
# #         "is_dae": True,
# #         "nx": nx,
# #         "nu": nu,
# #         "nz": nz,
# #         "Z_NAMES": Z_NAMES,
# #         "F_all": F_all,
# #     }


















# MODEL MISMATCH

# # -------------------------------
# # 1. Model infeasibility
# # ------------------------------
#
# if include_model_mismatch:
#     if z_model_list is None:
#         raise ValueError("z_model_list must not be None")
#
#     scale = np.array([
#         1 / 10,  # P_bh_bar
#         1 / 10,  # P_tb_b_bar
#         1.0,  # w_G_inj
#         1.0,  # w_res
#         1.0,  # w_L_res
#         1.0,  # w_G_res
#         1.0,  # w_w_out
#         1.0,  # w_o_out
#     ], dtype=float)
#
#     for z_p_j, z_m_j in zip(z_plant_list, z_model_list):
#         err_j = np.array(z_p_j).reshape((-1,)) - np.array(z_m_j).reshape((-1,))
#         err_j_scaled = scale * err_j
#
#         theta_model_j = float(np.linalg.norm(err_j_scaled, ord=2))
#
#         theta_vector.append(theta_model_j)
#         theta_details["model_mismatch_per_well"].append(theta_model_j)



#Garbage dumper

# import casadi as ca
# from application.simulation_engine import make_model
# from utilities.block_builders import *
# from networks.networks import *
#
#
# def optimize_single_well_production_NN(
#         F_u2z,
#         F_pinn,
#         u_guess=(0.5,0.5),
#         P_max_tb_b_bar=120,
#         P_min_bh_bar=90,
#         ):
#     ipopt_opts = {
#         "ipopt.print_level": 0,
#         "print_time": 0,
#         "ipopt.max_iter": 6000,
#         "ipopt.tol": 1e-10,
#         "ipopt.constr_viol_tol": 1e-8,
#         "ipopt.mu_strategy": "adaptive",
#         "ipopt.linear_solver": "mumps",
#     }
#
#     # ---------------------
#     # Decision variable (single well)
#     # ---------------------
#     u = ca.MX.sym("u", 2)  # [u1,u2]
#
#     u1 = u[0]
#     u2 = u[1]
#
#     # Evaluate NN surrogate
#     z = F_u2z(u=u)["z"] if isinstance(F_u2z(u=u), dict) else F_u2z(u)  # robust
#     m_o_out = z[0]
#     P_bh = z[1]
#     P_tb_b = z[2]
#
#     # ---------------------
#     # Objective: maximize oil
#     # ---------------------
#     obj = -m_o_out
#
#     # ---------------------
#     # Constraints
#     #   P_bh >= P_min
#     #   P_tb_b <= P_max
#     # ---------------------
#     b_hat = -0.3268*u1*u1 + 0.5116*u1 + 0.01914
#     g_stab = u2 - b_hat
#
#
#     g = ca.vertcat(P_bh, P_tb_b,g_stab)
#     lbg = ca.DM([float(P_min_bh_bar), -ca.inf,0.0])
#     ubg = ca.DM([ca.inf, float(P_max_tb_b_bar),ca.inf])
#
#     # ---------------------
#     # Bounds / initial guess
#     # ---------------------
#     lbx = ca.DM([0.05, 0.10])
#     ubx = ca.DM([1.0, 1.0])
#     x0 = ca.DM(list(u_guess))
#
#     # ---------------------
#     # Solve NLP
#     # ---------------------
#     nlp = {"x": u, "f": obj, "g": g}
#     solver = ca.nlpsol("single_well_solver", "ipopt", nlp, ipopt_opts)
#
#     sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
#     stats = solver.stats()
#
#     u_star = ca.DM(sol["x"]).full().flatten()
#     y_star = ca.DM(F_pinn(u=u_star)["y"]).full().flatten()
#     z_star = ca.DM(F_u2z(u=u_star)["z"] if isinstance(F_u2z(u=u_star), dict) else F_u2z(u_star)).full().flatten()
#
#     return {
#         "stats": stats,
#         "u_star": u_star,  # [u1*, u2*]
#         "y_star": y_star,
#         "z_star": z_star,  # [m_o_out*, P_bh*, P_tb_b*]
#         "m_o_out": float(z_star[0]),
#         "P_bh": float(z_star[1]),
#         "P_tb_b": float(z_star[2]),
#     }
# #
# #
# from pathlib import Path
# from configuration.wells import get_wells
#
#
#
#
#
#
#
#
# ### Change well name here
# well_name="P2"
# ###########################
# wells=get_wells()
# well_list=wells[well_name]
#
#
# pinn_path, alg_path=get_model_paths(well_name)
#
# BSW=well_list["BSW"]
# GOR=well_list["GOR"]
# PI=well_list["PI"]
# K_gs_sur=well_list["K_gs_sur"]
# K_inj_sur=well_list["K_inj_sur"]
# K_pr_sur=well_list["K_pr_sur"]
# y_guess_sur=well_list["y_guess_sur"]
# y_min=well_list["y_min"]
# y_max=well_list["y_max"]
# z_min=well_list["z_min"]
# z_max=well_list["z_max"]
#
#
# pinn = PINN(hidden_units=[64,64,64],
#             y_min=y_min,
#             y_max=y_max)
#
#
# algnn = AlgNN(hidden_units=[64,64,64,64],
#               y_min=y_min,
#               y_max=y_max,
#               z_min=z_min,
#               z_max=z_max)
#
#
# # 1) Load your torch models
# # from your_code import PINN, AlgNN, load_model_weights
# pinn = load_model_weights(pinn, pinn_path)
# algnn  = load_model_weights(algnn, alg_path)
#
# # 2) Extract weights
# pinn_w = extract_pinn_standard_weights(pinn)
# alg_w  = extract_algnn_standard_weights(algnn)
#
# # 3) IMPORTANT: use the SAME scaling constants as the trained models
# # You can read them from buffers:
# pinn_y_min = pinn.y_min.cpu().numpy().tolist()
# pinn_y_max = pinn.y_max.cpu().numpy().tolist()
# pinn_u_min = pinn.u_min.cpu().numpy().tolist()
# pinn_u_max = pinn.u_max.cpu().numpy().tolist()
#
# alg_y_min = algnn.y_min.cpu().numpy().tolist()
# alg_y_max = algnn.y_max.cpu().numpy().tolist()
# alg_u_min = algnn.u_min.cpu().numpy().tolist()
# alg_u_max = algnn.u_max.cpu().numpy().tolist()
# alg_z_min = algnn.z_min.cpu().numpy().tolist()
# alg_z_max = algnn.z_max.cpu().numpy().tolist()
#
# F_u2z = build_casadi_surrogate_u2z(
#     pinn_weights=pinn_w,
#     algnn_weights=alg_w,
#     pinn_y_min=pinn_y_min, pinn_y_max=pinn_y_max,
#     pinn_u_min=pinn_u_min, pinn_u_max=pinn_u_max,
#     alg_y_min=alg_y_min,   alg_y_max=alg_y_max,
#     alg_u_min=alg_u_min,   alg_u_max=alg_u_max,
#     alg_z_min=alg_z_min,   alg_z_max=alg_z_max,
#     BSW=BSW,
#     GOR=GOR,
# )
#
# # # F_u2z is your chained CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
# # res = optimize_single_well_production_NN(
# #     F_u2z=F_u2z,
# #     F_pinn=F_pinn,
# #     u_guess=(0.8, 0.8),
# #     P_min_bh_bar=90.0,
# #     P_max_tb_b_bar=120.0,
# # )
# #
# # print("Solver success:", res["stats"]["success"])
# # print("u* =", res["u_star"])
# # print("y* =", res["y_star"])
# # print("m_o_out* =", res["m_o_out"])
# # print("P_bh* =", res["P_bh"])
# # print("P_tb_b* =", res["P_tb_b"])
# #
# # from solvers.steady_state_solver import solve_equilibrium_ipopt
# # u1=res["u_star"][0]
# # u2=res["u_star"][1]
# #
# # y_guess=res["y_star"]
# #
# # y_guess_rig = [3679.08033973,
# #                289.73390193,
# #                3167.56224658,
# #                1041.96126532,
# #                50.46858403,
# #                759.52720527,
# #                249.84447542]
# #
# # z_guess_rig = [8.75897957e+06,
# # 8.42155186e+06,
# # 2.17230613e+01,
# # 2.17230613e+01]
# #
# # print('\n\n')
# # print(f"Now we take this control and apply to the plant...")
# # model=make_model("rigorous",BSW=0.20,GOR=0.05,PI=3.0e-6)
# # y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
# #     model=model,
# #     u_val=[u1, u2],
# #     y_guess=y_guess_rig,
# #     z_guess=z_guess_rig
# # )
# # print(f"y_star of the model used to train this NN at u*={y_star}")
# # print(f"And the pressure bottomhole is p_bh={out_star[15]} and w_out={out_star[38]}")



# def flatten_sweep_results_to_batch_full(results: dict, only_success: bool = True):
#     """
#     Like flatten_sweep_results_to_batch, but also returns z targets from OUT.
#
#     Assumes:
#       - first 3 Z_NAMES are states y = [y1,y2,y3]
#       - next 3 Z_NAMES are targets z = [p_bh_bar, p_tb_b_bar] (or whatever order is in Z_NAMES)
#     """
#     import numpy as np
#     import torch
#
#     u1_grid = np.asarray(results["u1_grid"], dtype=float)
#     u2_grid = np.asarray(results["u2_grid"], dtype=float)
#
#     Z_NAMES = list(results["Z_NAMES"])
#     OUT = results["OUT"]
#
#     SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
#     RES_DX = np.asarray(results["RES_DX"], dtype=float)
#
#     Nu1 = len(u1_grid)
#     Nu2 = len(u2_grid)
#
#     # u grid
#     U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
#     u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)
#
#     # y (first 3)
#     y_cols = []
#     for name in Z_NAMES[:3]:
#         arr = np.asarray(OUT[name], dtype=float)
#         y_cols.append(arr.reshape(-1))
#     y_flat = np.stack(y_cols, axis=1)
#
#     # z targets (next 3)
#     z_cols = []
#     names=["P_bh_bar", "P_tb_b_bar"]
#     for name in names:
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
#         "u_np": u_np, "y_np": y_np, "z_np": z_np,
#         "u_t": u_t, "y_t": y_t, "z_t": z_t,
#         "res_dx_t": res_dx_t,
#         "Nu1": Nu1, "Nu2": Nu2,
#         "mask_np": mask,
#     }

# if __name__ == "__main__":
#     N_data = 20  # -> total grid points = 100
#
#     wells=get_wells()
#     well="P2"
#
#     BSW = wells[well]["BSW"]
#     GOR = wells[well]["GOR"]
#     PI = wells[well]["PI"]
#     K_gs = wells[well]["K_gs_sur"]
#     K_inj = wells[well]["K_inj_sur"]
#     K_pr = wells[well]["K_pr_sur"]
#     y_guess_sur = wells[well]["y_guess_sur"]
#
#     # Choose your bounds (edit these!)
#     u1_min, u1_max = 0.05, 1.00
#     u2_min, u2_max = 0.10, 1.00
#
#     # Your initial guess for the solver (edit if needed)
#     # Must match model nx (for your surrogate it's 3)
#
#     y_guess_sur = np.array(y_guess_sur, dtype=float)
#     RES_TOL_DX=1e-6
#     RES_TOL_G=1e-6
#     TOL_EIG=1e-8
#     print("Running sweep...")
#     results = build_and_run_surrogate_sweep(
#         u1_min=u1_min,
#         u2_min=u2_min,
#         N_data=N_data,
#         y_guess_init=y_guess_sur,
#         BSW=BSW,
#         GOR=GOR,
#         PI=PI,
#         K_gs=K_gs,
#         K_inj=K_inj,
#         K_pr=K_pr
#     )
#
#     print("\nFlattening to batch...")
#     batch = flatten_sweep_results_to_batch(results,
#                                            only_success=True)
#
#     device='cpu'
#
#     y_np = batch["y_np"]
#
#     y_min = np.min(y_np, axis=0)
#     y_max = np.max(y_np, axis=0)
#
#     margin = 0.15  # 15%
#
#     y_span = y_max - y_min
#     y_min_loose = y_min - margin * y_span
#     y_max_loose = y_max + margin * y_span
#
#     print("\n--- State ranges (loose) ---")
#     for i, name in enumerate(batch["Z_NAMES"][:3]):
#         print(f"{name}: min = {y_min_loose[i]:.6f}, max = {y_max_loose[i]:.6f}")
#
#
#     u = batch["u_t"].to(device)
#     y = batch["y_t"].to(device)
#     # z = batch["z_t"].to(device)  # <- ground truth for AlgNN
#     print("Z_NAMES:", batch["Z_NAMES"][:3])
#     print(u.shape, y.shape)
#
#     print("\n--- Sweep summary ---")
#     Nu1, Nu2 = batch["Nu1"], batch["Nu2"]
#     total = Nu1 * Nu2
#     success_total = int(np.sum(results["SUCCESS"]))
#     print(f"Grid: {Nu1} x {Nu2} = {total} points")
#     print(f"SUCCESS count (raw): {success_total}")
#     print(f"Batch size (finite OUT & success): {batch['u_np'].shape[0]}")
#     print(f"Z_NAMES: {batch['Z_NAMES']}")
#
#     # Torch shapes
#     print("\n--- Torch tensors ---")
#     print("u_t:", tuple(batch["u_t"].shape), batch["u_t"].dtype)
#     print("res_dx_t:", tuple(batch["res_dx_t"].shape), batch["res_dx_t"].dtype)
#
#     # Print a few samples
#     print("\n--- First all samples ---")
#     for k in range(batch["u_np"].shape[0]):
#         u_k = batch["u_np"][k]
#         rd_k = batch["res_dx_np"][k]
#         y_k = batch["y_np"][k]
#         print(f"{k:02d} | u={u_k}  | res_dx={rd_k:.2e} | y={y_k} ")


######################################################

#
#
# def flatten_sweep_results_to_batch(results: dict,
#                                    only_success: bool = True):
#     """
#     Convert run_sweep() output dict into a flat batch.
#
#     Uses ONLY:
#       - u = [u1, u2]
#       - y = first 3 entries of Z_NAMES (assumed states)
#       - success mask (stable is ignored)
#
#     Returns:
#       u_np : (N,2)
#       y_np : (N,3)
#       success_np : (N,) bool
#       res_dx_np : (N,) float
#       u_t : torch.FloatTensor (N,2)
#       y_t : torch.FloatTensor (N,3)
#       res_dx_t : torch.FloatTensor (N,)
#     """
#     import numpy as np
#     import torch
#
#     u1_grid = np.asarray(results["u1_grid"], dtype=float)
#     u2_grid = np.asarray(results["u2_grid"], dtype=float)
#
#     Z_NAMES = list(results["Z_NAMES"])
#     OUT = results["OUT"]
#
#     SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
#     RES_DX = np.asarray(results["RES_DX"], dtype=float)
#
#     Nu1 = len(u1_grid)
#     Nu2 = len(u2_grid)
#
#     # Build u grid (Nu1,Nu2) -> (Nu1*Nu2,2)
#     U1=results["U1"]
#     U2=results["U2"]
#
#     # U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="ij")
#     u_flat = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)
#
#     # Extract ONLY first 3 outputs (states) in Z_NAMES order
#     if len(Z_NAMES) < 3:
#         raise ValueError(f"Expected at least 3 Z_NAMES entries, got {len(Z_NAMES)}")
#
#     y_cols = []
#     for name in Z_NAMES[:3]:
#         arr = np.asarray(OUT[name], dtype=float)   # (Nu1,Nu2)
#         y_cols.append(arr.reshape(-1))
#     y_flat = np.stack(y_cols, axis=1)  # (Nu1*Nu2, 3)
#
#     success_flat = SUCCESS.reshape(-1)
#     res_dx_flat = RES_DX.reshape(-1)
#
#     # Valid rows: success AND finite y
#     finite_y = np.all(np.isfinite(y_flat), axis=1)
#     mask = success_flat & finite_y if only_success else finite_y
#
#     u_np = u_flat[mask]
#     y_np = y_flat[mask]
#     success_np = success_flat[mask]
#     res_dx_np = res_dx_flat[mask]
#
#     # Torch tensors
#     u_t = torch.tensor(u_np, dtype=torch.float32)
#     y_t = torch.tensor(y_np, dtype=torch.float32)
#     res_dx_t = torch.tensor(res_dx_np, dtype=torch.float32)
#
#     return {
#         "Z_NAMES": Z_NAMES,
#         "u_np": u_np,
#         "y_np": y_np,
#         "success_np": success_np,
#         "res_dx_np": res_dx_np,
#         "u_t": u_t,
#         "y_t": y_t,
#         "res_dx_t": res_dx_t,
#         "Nu1": Nu1,
#         "Nu2": Nu2,
#         "mask_np": mask,
#     }
#

#
#
#     u1_grid_train = np.linspace(float(self.u_min_train[0]), float(self.u_max_train[0]) + 1e-5, int(self.N_data))
#     u2_grid_train = np.linspace(float(self.u_min_train[1]), float(self.u_max_train[1]) + 1e-5, int(self.N_data))
#
#     model_sur = make_model("surrogate",
#                            BSW=self.BSW,
#                            GOR=self.GOR,
#                            PI=self.PI,
#                            K_gs=self.K_gs,
#                            K_inj=self.K_inj,
#                            K_pr=self.K_pr)
#
#     results_train = run_sweep(
#         model_sur,
#         U1_MIN=self.u1_min,
#         U2_MIN=self.u2_min,
#         U_SIM_SIZE=self.N_data,
#         y_guess_init=self.y_guess_surr)
#
#
#     batch_train = flatten_sweep_results_to_batch(results_train, only_success=True)
#
#     # Move to device
#     self.u_data_train = batch_train["u_t"].to(self.device)
#     self.y_data_train = batch_train["y_t"].to(self.device)
#
#     # ---------------------------------
#     # 3) Supervised sweep dataset (VAL)
#     # ---------------------------------
#     N_data_val = int(self.N_data / 3)
#     N_data_val = max(3, N_data_val)  # keep a minimum grid so it isn't degenerate
#
#     self.l.info(f"Generating VAL sweep dataset with N_data_val={N_data_val} (grid {N_data_val}x{N_data_val}) ...")
#
#     u1_grid_val = np.linspace(self.u_min_train[0], self.u_max_train[0] + 1e-5, N_data_val, dtype=float)
#     u2_grid_val = np.linspace(self.u_min_train[1], self.u_max_train[1] + 1e-5, N_data_val, dtype=float)
#
#     results_val = run_sweep(
#         model_sur,
#         U1_MIN=self.u1_min,
#         U2_MIN=self.u2_min,
#         U_SIM_SIZE=self.N_data,
#         y_guess_init=self.y_guess_surr,
#     )
#
#     batch_val= flatten_sweep_results_to_batch(results_val, only_success=True)
#
#     # Move to device
#     self.u_data_val = batch_val["u_t"].to(self.device)
#     self.y_data_val = batch_val["y_t"].to(self.device)
#
#     self.y_data_train_norm = (self.y_data_train - self.y_min_t) / (self.y_range_t)
#     self.print_shapes()
#
#
# def print_shapes(self):
#     self.l.info(f"Size of self y data train is:{self.y_data_train.shape}")
#     self.l.info(f"Size of self u data train is:{self.u_data_train.shape}")
#
#     self.l.info(f"Size of self y data val is:{self.y_data_val.shape}")
#
#     self.l.info(f"Size of the colocation poins is {self.u_col.shape}")
# # after loss.backward()
# with torch.no_grad():
#     # 1) grad norm
#     g2 = 0.0
#     for p in self.net.parameters():
#         if p.grad is not None:
#             g2 += p.grad.detach().float().norm().item() ** 2
#     grad_norm = g2 ** 0.5
#
#     # 2) step size proxy: parameter norm change if we took a step (approx)
#     lr = self._optim.param_groups[0]["lr"]
#     self.l.info(f"[DBG] epoch={self._e} grad_norm={grad_norm:.3e} lr={lr:.3e}")
# with torch.no_grad():
#     w2 = 0.0
#     for p in self.net.parameters():
#         w2 += p.detach().float().norm().item() ** 2
#     w_norm = w2 ** 0.5
#     self.l.info(f"[DBG] epoch={self._e} w_norm={w_norm:.6e}")
# #STAGE 3 - L-BFGS Block 2
# self.l.info(f"--- Starting Stage 3: L-BFGS Block 2 (Adam Epochs: {self._e}) ---")
# # We re-initialize the optimizer to run a new session
# self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_loop)
#
# data_to_log, val_score = self._run_epoch()
# self._e += lbfgs_iter_per_loop  # Log this as 1000 more "epochs"
#
# if self._log_to_wandb:
#     wandb.log(data_to_log, step=self._e, commit=True)
#     self.save_checkpoint()
# if val_score < self.best_val:
#     self.best_val = val_score
#     self.train_loss_at_best_val = data_to_log['train/total']
#     if self._log_to_wandb:
#         self.save_model(name='model_best')
#
# # --- FINISH ---

# def _debug_grad_norm(self):
#     total = 0.0
#     count = 0
#     max_g = 0.0
#     for p in self.net.parameters():
#         if p.grad is None:
#             continue
#         g = p.grad.detach()
#         n = torch.norm(g).item()
#         total += n
#         max_g = max(max_g, g.abs().max().item())
#         count += 1
#     return {"grad_norm_sum": total, "grad_absmax": max_g, "grad_tensors": count}

# def _print_one_time_sanity(self, y_hat_col, dx_col, y_pred_data, loss_data_n, loss_data_raw):
#     # 1) output scale sanity
#     with torch.no_grad():
#         self.l.info("[SANITY] ---- One-time debug ----")
#         self.l.info(f"[SANITY] y_hat_col mean={y_hat_col.mean(0).detach().cpu().numpy()} "
#                     f"std={y_hat_col.std(0).detach().cpu().numpy()} "
#                     f"min={y_hat_col.min(0).values.detach().cpu().numpy()} "
#                     f"max={y_hat_col.max(0).values.detach().cpu().numpy()}")
#         self.l.info(f"[SANITY] dx_col mean={dx_col.mean(0).detach().cpu().numpy()} "
#                     f"std={dx_col.std(0).detach().cpu().numpy()} "
#                     f"mse_comp={torch.mean(dx_col * dx_col, dim=0).detach().cpu().numpy()}")
#
#         self.l.info(f"[SANITY] y_data mean={self.y_data.mean(0).detach().cpu().numpy()} "
#                     f"std={self.y_data.std(0).detach().cpu().numpy()} "
#                     f"min={self.y_data.min(0).values.detach().cpu().numpy()} "
#                     f"max={self.y_data.max(0).values.detach().cpu().numpy()}")
#
#         self.l.info(f"[SANITY] y_pred_data mean={y_pred_data.mean(0).detach().cpu().numpy()} "
#                     f"std={y_pred_data.std(0).detach().cpu().numpy()} "
#                     f"min={y_pred_data.min(0).values.detach().cpu().numpy()} "
#                     f"max={y_pred_data.max(0).values.detach().cpu().numpy()}")
#
#         self.l.info(f"[SANITY] data_norm={float(loss_data_n.item()):.3e} "
#                     f"data_raw={float(loss_data_raw.item()):.3e} "
#                     f"lambda_data={self.lambda_data}")
#
#         # probe
#         y_probe, dx_probe, dxn = self._debug_probe()
#         self.l.info(
#             f"[SANITY] u_probe={tuple(self.u_probe.tolist())} y_hat={y_probe} dx_probe={dx_probe} ||dx||={dxn:.3e}")
#
#         # last layer stats
#         stats = self._debug_param_and_grad_stats()
#         if stats:
#             self.l.info(f"[SANITY] last layer stats: {stats}")
#         self.l.info("[SANITY] -----------------------")
#
# def _debug_check_ranges_and_nans(self):
#     # Check u ranges
#     def _rng(t: torch.Tensor):
#         return t.min(dim=0).values.detach().cpu().numpy(), t.max(dim=0).values.detach().cpu().numpy()
#
#     ucol_min, ucol_max = _rng(self.u_col)
#     uval_min, uval_max = _rng(self.u_val)
#     udat_min, udat_max = _rng(self.u_data)
#
#     self.l.info(f"[CHECK] u_col min={ucol_min} max={ucol_max}")
#     self.l.info(f"[CHECK] u_val min={uval_min} max={uval_max}")
#     self.l.info(f"[CHECK] u_data min={udat_min} max={udat_max}")
#     self.l.info(f"[CHECK] expected u_min={self.u_min_train} u_max={self.u_max_train}")
#
#     # Check y ranges
#     y_min = self.y_data.min(dim=0).values.detach().cpu().numpy()
#     y_max = self.y_data.max(dim=0).values.detach().cpu().numpy()
#     self.l.info(f"[CHECK] y_data min={y_min} max={y_max}")
#     self.l.info(f"[CHECK] y_mu={self.y_mu.detach().cpu().numpy().reshape(-1)}")
#     self.l.info(f"[CHECK] y_std={self.y_std.detach().cpu().numpy().reshape(-1)}")
#
#     # NaN/inf checks
#     def _finite(name, t):
#         ok = torch.isfinite(t).all().item()
#         self.l.info(f"[CHECK] {name} finite={ok}")
#         if not ok:
#             bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:10].detach().cpu().numpy()
#             self.l.info(f"[CHECK] {name} first bad idx: {bad}")
#
#     _finite("u_col", self.u_col)
#     _finite("u_val", self.u_val)
#     _finite("u_data", self.u_data)
#     _finite("y_data", self.y_data)
#     _finite("res_dx_data", self.res_dx_data)
#
# @torch.no_grad()
# def _debug_probe(self):
#     self.net.eval()
#     y_probe = self.net(self._u_probe_t)
#     dx_probe = self.physics_f(y_probe, self._u_probe_t)
#     return (
#         y_probe.detach().cpu().numpy().reshape(-1),
#         dx_probe.detach().cpu().numpy().reshape(-1),
#         float(torch.norm(dx_probe).item()),
#     )
#
# def _debug_param_and_grad_stats(self):
#     # Params: last layer is usually where scale comes from
#     last = None
#     for m in self.net.modules():
#         if isinstance(m, torch.nn.Linear):
#             last = m
#     if last is None:
#         return {}
#
#     with torch.no_grad():
#         w = last.weight
#         b = last.bias
#         stats = {
#             "last_w_norm": float(torch.norm(w).item()),
#             "last_b_norm": float(torch.norm(b).item()) if b is not None else 0.0,
#             "last_w_absmax": float(w.abs().max().item()),
#             "last_b_absmax": float(b.abs().max().item()) if b is not None else 0.0,
#         }
#     return stats


# self.l.info("Loading full sweep dataset for validation from:")
        # self.l.info(str(self.sweep_results_path))
        #
        # if not self.sweep_results_path.exists():
        #     raise FileNotFoundError(
        #         f"Could not find sweep_results.pkl for well {self.well_name}: "
        #         f"{self.sweep_results_path}"
        #     )
        #
        # with open(self.sweep_results_path, "rb") as f:
        #     sweep_results = pickle.load(f)
        #
        # output_names = poly_dataset["pinn_output_names"]
        #
        # u1_grid = np.asarray(sweep_results["u1_grid"], dtype=float)
        # u2_grid = np.asarray(sweep_results["u2_grid"], dtype=float)
        #
        # u_val_list = []
        # y_val_list = []
        #
        # for i, u1 in enumerate(u1_grid):
        #     for j, u2 in enumerate(u2_grid):
        #
        #         y_row = [
        #             sweep_results["OUT"][name][i, j]
        #             for name in output_names
        #         ]
        #
        #         y_row = np.asarray(y_row, dtype=float)
        #
        #         if np.all(np.isfinite(y_row)):
        #             u_val_list.append([u1, u2])
        #             y_val_list.append(y_row)
        #
        # u_data_val_np = np.asarray(u_val_list, dtype=np.float32)
        # y_data_val_np = np.asarray(y_val_list, dtype=np.float32)
        #
        # self.u_data_val = torch.tensor(
        #     u_data_val_np,
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        #
        # self.y_data_val = torch.tensor(
        #     y_data_val_np,
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        #
        # self.y_data_val_norm = (self.y_data_val - self.y_min_t) / self.y_range_t

@torch.no_grad()
def compute_physics_residual_scan(self, u_scan, y_ref=None, batch_size=2048):
    """
    Evaluate PINN physics residual over a full grid.

    Returns a dictionary with:
        u
        y_pred
        dx
        dx_abs
        dx_norm
        point_loss
        bad_mask
    """

    self.net.eval()

    u_all = []
    y_all = []
    dx_all = []

    for i in range(0, u_scan.shape[0], batch_size):
        u_b = u_scan[i:i + batch_size]
        y_b = self.net(u_b)

        physics_out = self.physics_f(
            y_b,
            u_b,
            BSW=self.BSW,
            GOR=self.GOR,
            PI=self.PI,
            K_gs=self.K_gs,
            K_inj=self.K_inj,
            K_pr=self.K_pr,
        )

        if isinstance(physics_out, tuple):
            dx_b = physics_out[0]
        else:
            dx_b = physics_out

        u_all.append(u_b.detach().cpu())
        y_all.append(y_b.detach().cpu())
        dx_all.append(dx_b.detach().cpu())

    u_all = torch.cat(u_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    dx_all = torch.cat(dx_all, dim=0)

    dx_abs = torch.abs(dx_all)
    dx_norm = torch.linalg.norm(dx_all, dim=1)

    # Pointwise raw residual MSE, not scaled by y_range.
    point_loss = torch.mean(dx_all ** 2, dim=1)

    return {
        "u": u_all,
        "y_pred": y_all,
        "dx": dx_all,
        "dx_abs": dx_abs,
        "dx_norm": dx_norm,
        "point_loss": point_loss,
    }













