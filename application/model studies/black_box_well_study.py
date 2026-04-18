# GENERAL LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt

# SPECIFIC MODULES
from application.simulation_engine import make_model, run_sweep, fit_boundary_polynomial, extract_stability_boundary_from_grid
from application.plotting_engine import plot_stability_map, overlay_boundary_and_fit, plot_w_o_out_contour, plot_surface

#Wells
from configuration.wells import get_wells

wells=get_wells()


######################################################################################
############### THIS HERE IS NEVE GOING TO CHANGE ####################################
MODE = "rigorous"
RES_TOL_DX = 1e-6 # This could be a global variable
RES_TOL = 1e-6
RES_TOL_G = 1e-6 # This could be a global variable
TOL_EIG = 1e-8
small_delta=1e-2
U1_MIN=0.05 #if you ever change this, you need to change inside the black box optimizer
U2_MIN=0.10 #if you ever change this, you need to change inside the black box optimizer
U_SIM_SIZE=20
degree_polynomial = 2
######################################################################################
######################################################################################



######################################################################################
### PLEASE INPUT HERE WHAT KIND OF PLOTS DO YOU WANT #################################
all_plots=False #TBB

stability_map=True
stability_plot=True

focused_figures=True
heatmap_w_out=True
######################################################################################
######################################################################################


results_all_wells = {}
coeff_stability={}


for well, params in wells.items():
    print("\n" + "=" * 80)
    print(f"RUNNING WELL {well}")
    print("=" * 80)

    BSW = params["BSW"]
    GOR = params["GOR"]
    PI = params["PI"]
    K_gs = params["K_gs"]
    K_inj = params["K_inj"]
    K_pr = params["K_pr"]
    y_guess_rig = params["y_guess_rig"]
    z_guess_rig = params["z_guess_rig"]

    model_rig = make_model("rigorous",
                       BSW=BSW,
                       GOR=GOR,
                       PI=PI,
                       K_gs=K_gs,
                       K_inj=K_inj,
                       K_pr=K_pr)

    y_guess = np.array(y_guess_rig, dtype=float).reshape(-1)
    if y_guess.size != model_rig["nx"]:
        raise ValueError(f"y_guess_init has size {y_guess.size}, but model nx={model_rig["nx"]}")

    results_all_wells[well] = run_sweep(model_rig,
                                  U1_MIN=U1_MIN,
                                  U2_MIN=U2_MIN,
                                  U_SIM_SIZE=U_SIM_SIZE,
                                  y_guess_init=y_guess_rig,
                                  z_guess_init=z_guess_rig,
                                  RES_TOL_DX=RES_TOL_DX,
                                  TOL_EIG=TOL_EIG)

    if stability_map:
        boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(
            results_all_wells[well]["u1_grid"],
            results_all_wells[well]["u2_grid"],
            results_all_wells[well]["STABLE"]
        )
        b_hat=fit_boundary_polynomial(boundary_u1,
                                      boundary_u2,
                                      deg=degree_polynomial)
        print(boundary_u1, boundary_u2)
        print (b_hat)
        coeff_stability[well] = {
            "coef": b_hat.c,
            "poly": b_hat
        }
        if stability_plot:
                ax=plot_stability_map(U1=results_all_wells[well]["U1"],
                                   U2=results_all_wells[well]["U2"],
                                   STABLE=results_all_wells[well]["STABLE"],
                                   U1_MIN=U1_MIN,
                                   U2_MIN=U2_MIN,
                                   title=f"Mapa de estabilidade para o poço {well}")
                overlay_boundary_and_fit(ax,
                                         b_hat)
                plt.tight_layout()

                save_dir = os.path.join("results", "figures", "stability_maps")
                os.makedirs(save_dir, exist_ok=True)

                filename = f"stability_map_{well}.pdf"
                filepath = os.path.join(save_dir, filename)

                ax.figure.savefig(filepath, format="pdf", bbox_inches="tight")
                plt.show()

    ax=plot_w_o_out_contour(results_all_wells[well],
                         title=f"Produção de óleo do poço {well}",
                         only_stable=True,
                         only_success=True,
                         levels=40)
    ax.figure.tight_layout()
    ax.figure.show()

    if focused_figures:
        U1=results_all_wells[well]["U1"]
        U2=results_all_wells[well]["U2"]

        fig1 = plt.figure(figsize=(9,9))
        fig1.suptitle("Variáveis de interesse para definição do problema de otimização", fontsize=20)

        axes = [
            fig1.add_subplot(2, 2, 1, projection="3d"),
            fig1.add_subplot(2, 2, 2, projection="3d"),
            fig1.add_subplot(2, 2, 3, projection="3d"),
            fig1.add_subplot(2, 2, 4, projection="3d")
        ]
        plot_surface(fig1,axes[0], U1, U2, results_all_wells[well]["OUT"]["P_bh_bar"], "Bottomhole Pressure", "Bottomhole Pressure (bar)")
        plot_surface(fig1, axes[1], U1, U2, results_all_wells[well]["OUT"]["P_an_bar"], "Annulus Pressure", "Annulus Pressure(bar)")
        plot_surface(fig1,axes[2], U1, U2, results_all_wells[well]["OUT"]["P_tb_b_bar"],  "Tubing Pressure", "Tubing Pressure (bar)")
        plot_surface(fig1,axes[3], U1, U2, results_all_wells[well]["OUT"]["w_G_inj"], "Gas Lift", "Gas Lift (kg/s)")
        axes[0].view_init(elev=40, azim=45)
        axes[1].view_init(elev=40,azim=45)# opposite view
        axes[2].view_init(elev=40,azim=45)# opposite view
        axes[3].view_init(elev=40,azim=45)# opposite view
        plt.tight_layout()
        plt.show()

print(f"coeff_stability: {coeff_stability}")




        ####################################
        # The usage later on here is the following:
        # def g_stability(u, b_hat):
        #     u1 = u[0]
        #     u2 = u[1]
        #     return b_hat(u1) - u2
        ####################################
#     plot_w_o_out_contour(results_all["rigorous"])
#     plt.tight_layout()
#     plt.show()
#
#
# elif MODE == "surrogate":
#     for var, title, zlabel in PLOT_VARS:
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(1, 1, 1, projection="3d")
#         plot_surface(fig, ax, U1, U2, results_all["surrogate"]["OUT"][var], title, zlabel)
#         ax.view_init(elev=25, azim=45)
#         plt.tight_layout()
#         plt.show()
#     plot_figures(results_all["surrogate"], title="surrogate")
#     ax = plot_stability_map(U1,
#                             U2,
#                             STABLE=results_all["surrogate"]["STABLE"],
#                             title="Stability map + fitted stability constraint")
#     boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(u1_grid, u2_grid,
#                                                                     results_all["surrogate"]["STABLE"])
#     b_hat = fit_boundary_polynomial(boundary_u1, boundary_u2)
#     print(boundary_u1, boundary_u2)
#     print(b_hat)
#     overlay_boundary_and_fit(ax, b_hat, deg=2)
#     plt.tight_layout()
#     plt.show()
#
#     print(f"Minimum m_G_an is {np.min(results_all["surrogate"]["OUT"]["m_G_an"])}")
#     print(f"Minimum m_G_tb is {np.min(results_all["surrogate"]["OUT"]["m_G_t"])}")
#     print(f"Minimum m_L_tb is {np.min(results_all["surrogate"]["OUT"]["m_o_t"])}")
#
#     print(f"Maximum m_G_an is {np.max(results_all["surrogate"]["OUT"]["m_G_an"])}")
#     print(f"Maximum m_G_tb is {np.max(results_all["surrogate"]["OUT"]["m_G_t"])}")
#     print(f"Maximum m_L_tb is {np.max(results_all["surrogate"]["OUT"]["m_o_t"])}")
#
#     print(f"Minimum P_bh_bar is {np.min(results_all["surrogate"]["OUT"]["P_bh_bar"])}")
#     print(f"Minimum P_tb_b_bar is {np.min(results_all["surrogate"]["OUT"]["P_tb_b_bar"])}")
#     print(f"Minimum mass of oil produced is {np.min(results_all["surrogate"]["OUT"]["w_o_out"])}")
#
#     print(f"MaximumP_bh_bar is {np.max(results_all["surrogate"]["OUT"]["P_bh_bar"])}")
#     print(f"Maximum P_tb_b_bar is {np.max(results_all["surrogate"]["OUT"]["P_tb_b_bar"])}")
#     print(f"Maximum mass of oil produced is {np.max(results_all["surrogate"]["OUT"]["w_o_out"])}")
#
#
# elif MODE == "both":
#     for var, title, zlabel in PLOT_VARS:
#         if (var in results_all["rigorous"]["OUT"]) and \
#                 (var in results_all["surrogate"]["OUT"]):
#             plot_overlay_surface(
#                 var,
#                 U1,
#                 U2,
#                 results_all,
#                 title=title,
#                 zlabel=zlabel
#             )
#             # save using variable name
#             fig = plt.gcf()
#             fig.tight_layout()
#             fig.savefig("LAST_FIGURE.png", dpi=300, bbox_inches="tight")
#             plt.show()
#
#     plot_figures(results_all["rigorous"], title="Rigorous Model (DAE solver)")
#     plot_figures(results_all["surrogate"], title="Surrogate Model (PINN on the simplified simulation)")
#
#     ax = plot_stability_map(U1,
#                             U2,
#                             STABLE=results_all["rigorous"]["STABLE"],
#                             title="Stability map + fitted stability constraint for the Gas-Lift Rigorous Model")
#     boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(u1_grid, u2_grid,
#                                                                     results_all["rigorous"]["STABLE"])
#     b_hat = fit_boundary_polynomial(boundary_u1, boundary_u2)
#     print(boundary_u1, boundary_u2)
#     print(b_hat)
#     overlay_boundary_and_fit(ax, b_hat, deg=2)
#     plt.savefig("Stability plot.png", dpi=300, bbox_inches="tight")
#     plt.tight_layout()
#     plt.show()
#
#     # ax=plot_stability_map(U1,
#     #                       U2,
#     #                       STABLE=results_all["surrogate"]["STABLE"],
#     #                       title="Stability map + fitted stability constraint")
#     # boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(u1_grid, u2_grid, results_all["surrogate"]["STABLE"])
#     # b_hat=fit_boundary_polynomial(boundary_u1,boundary_u2)
#     # print(boundary_u1, boundary_u2)
#     # print (b_hat)
#     # overlay_boundary_and_fit(ax,b_hat,deg=2)
#     # plt.tight_layout()
#     # plt.show()