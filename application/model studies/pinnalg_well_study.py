"""
The objective of this file here, implemented for the article, is to generate a countour plot of
two wells for the article with PINNAlg for any output variable z.

Specifically for the article, we will do it for w_out, while also plotting constraints P_bh, P_tb, and stability.
A beautiful figure that will show the optimal for the two wells

Optionally:

- Compare PINN model with the surrogate model - This is just for debugging, as they should be equal or very close, but lets see
- Compare PINN model with the truth model - This shows the difference between models, we will not show this in the article
but can also be useful for appendix or results

"""
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import rotation

color_primary   = (54/255,  32/255, 229/255)   # blue
color_secondary = (240/255, 101/255, 74/255)   # orange
color_third     = (183/255, 53/255, 192/255)   # purple (saved)
from application.simulation_engine import make_model
from configuration.wells import get_wells

from application.plotting_engine import plot_contour, overlay_boundary_and_fit, overlay_common_boundaries
from application.simulation_engine import run_sweep, run_sweep_PINN, extract_threshold_boundary_from_grid,fit_boundary_polynomial

from utilities.block_builders import build_casadi_surrogate_u2z_for_well

import json

import numpy as np

#### Global values ###
U1_MIN = 0.05
U2_MIN = 0.10
U_SIM_SIZE = 100

z_guess_sur = None
RES_TOL_DX = 1e-6
RES_TOL_G = 1e-6
TOL_EIG = 1e-8
#####################

import matplotlib.ticker as ticker

################################ SELECT THE PLOTS YOU WANT #####################################
first_figure=False
second_figure=True
#################################################################################################


if first_figure:
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    wells = get_wells()
    sweeps=[]
    zmin = np.inf
    zmax = -np.inf



    for well, params in wells.items():
        well_name = well
        well_list = params

        model_NN = build_casadi_surrogate_u2z_for_well(well_name)

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

        j = 7

        U1_NN, U2_NN, Z_NN = run_sweep_PINN(
            model_NN,
            U1_MIN=U1_MIN,
            U2_MIN=U2_MIN,
            U_SIM_SIZE=U_SIM_SIZE
        )
        Zj=Z_NN[:, :, j]

        sweeps.append((well_name,U1_NN,U2_NN,Zj))

        zmin = min(zmin, np.nanmin(Zj))
        zmax = max(zmax, np.nanmax(Zj))

    fill_levels = np.linspace(zmin, zmax, 30)   # smooth color
    line_levels = np.linspace(zmin, zmax, 30)    # few clean lines

    fig,axes=plt.subplots(
        2, 1,
        figsize=(6.5, 8.5),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    mappable=None

    for ax, (well_name,U1_NN,U2_NN,Zj) in zip(axes,sweeps):
            u1_opt=wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("u")][0]
            u2_opt=wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("u")][1]
            z_opt=wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("w_o_out")]
            ax,cf = plot_contour(
                U1=U1_NN,
                U2=U2_NN,
                Z=Zj,
                u1_opt=u1_opt,
                u2_opt=u2_opt,
                z_opt=z_opt,
                title=f"$w_{{out,o}}$($u_1$,$u_2$) - Poço {well_name}",
                zlabel=names[j],
                fill_levels=fill_levels,
                line_levels=line_levels,
                mark_optimum=True,
                ax=ax,
                add_colorbar=False,
                vmin=zmin,
                vmax=zmax
            )
            mappable=cf
    # one shared colorbar
    # -------------------------
    # Create a dedicated colorbar axis
    # -------------------------
    cbar_ax = fig.add_axes([0.82, 0.10, 0.025, 0.8])
    # [left, bottom, width, height] in figure coordinates

    cbar = fig.colorbar(mappable, cax=cbar_ax)

    cbar.set_label(r"$w_{out,o} (kg/s)$", fontsize=16)

    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)
    plt.savefig("unconstrained_colormap.pdf", bbox_inches="tight",pad_inches=0.02)
    plt.show()

if second_figure:
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    opt_value=[11.07,16.47]

    U_SIM_SIZE = 100

    # ============================
    # 1. Load iteration sequence
    # ============================
    with open("u_sequence.json", "r") as f:
        u_sequence = np.array(json.load(f), dtype=float)
    with open("u_sequence_2.json", "r") as f:
        u_sequence_2 = np.array(json.load(f), dtype=float)

    print("u_sequence shape =", u_sequence.shape)   # expected: (n_iter, 4)


    u1_p1, u2_p1 = u_sequence[:, 0], u_sequence[:, 1]
    u1_p2, u2_p2 = u_sequence[:, 2], u_sequence[:, 3]


    u1_p1_sec, u2_p1_sec = u_sequence_2[:, 0], u_sequence_2[:, 1]
    u1_p2_sec, u2_p2_sec = u_sequence_2[:, 2], u_sequence_2[:, 3]



    U1_MIN = [np.min(u1_p1)-0.01,np.min(u1_p2)-0.01]
    U2_MIN = [np.min(u2_p1)-0.15,np.min(u2_p2)-0.15]
    U1_MAX = [np.max(u1_p1)+0.02, np.max(u1_p2)+0.02]
    U2_MAX = [np.max(u2_p1)+0.20,np.max(u2_p2)+0.10]

    print(u1_p1)

    # ============================
    # 2. Plot style
    # ============================
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    wells = get_wells()
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
    j = 7  # plot oil output

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8.5))

    # ============================
    # 3. Loop over wells
    # ============================
    for i, (ax, well_name) in enumerate(zip(axes, ["P1", "P2"])):
        well_list = wells[well_name]
        model_NN = build_casadi_surrogate_u2z_for_well(well_name)

        U1_NN, U2_NN, Z_NN = run_sweep_PINN(
            model_NN,
            U1_MIN=U1_MIN[i],
            U2_MIN=U2_MIN[i],
            U1_MAX=U1_MAX[i],
            U2_MAX=U2_MAX[i],
            U_SIM_SIZE=U_SIM_SIZE
        )

        Zj = Z_NN[:, :, j]

        u1_opt = well_list["optima"]["surrogate"]["total_unconstrained"]["u"][0]
        u2_opt = well_list["optima"]["surrogate"]["total_unconstrained"]["u"][1]
        z_opt = well_list["optima"]["surrogate"]["total_unconstrained"]["w_o_out"]

        ax, cf = plot_contour(
            U1=U1_NN,
            U2=U2_NN,
            Z=Zj,
            u1_opt=u1_opt,
            u2_opt=u2_opt,
            z_opt=z_opt,
            title=rf"$w_{{out,o}}(u_1,u_2)$ - Poço {well_name}",
            zlabel=names[j],
            fill_levels=10,
            line_levels=10,
            mark_optimum=False,
            ax=ax,
            add_colorbar=False,
            just_contour=True
        )
        ax.clabel(cf, cf.levels[::2],fmt="%.1f", inline=True, fontsize=12, rightside_up=True)
        # Dedicated colorbar for this subplot
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$w_{out,o}$ (kg/s)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Plot trajectory
        if well_name == "P1":
            uu1, uu2 = u1_p1, u2_p1
        else:
            uu1, uu2 = u1_p2, u2_p2

        ax.plot(uu1, uu2, marker="o", linewidth=1.8, color=color_primary)

        for k, (x, y) in enumerate(zip(uu1, uu2)):
            ax.annotate(
                str(k),
                (x, y),
                xytext=(-2, 5) if (k==0 or k==1 or k==2) else (-10,-18) if k==3 else (5,-15) if k==4 else (-2,6),
                textcoords="offset points",
                fontsize=14
            )


        u1_opt=uu1[5]
        u2_opt=uu2[5]
        opt=opt_value[i]
        label=f"Ótimo com restrições"

        ax.plot(
            u1_opt,
            u2_opt,
            "*",
            color=color_third,
            markersize=14,
            label=label
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc="upper center",
                frameon=True,
                fancybox=True,
                fontsize=12
            )
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_alpha(1.0)
            frame.set_linewidth(1.0)

    ax.set_xlim(U1_MIN[i], U1_MAX[i])
    ax.set_ylim(U2_MIN[i], U2_MAX[i])
    ax.set_xlabel(r"$u_1$",fontsize=16)
    ax.set_ylabel(r"$u_2$",fontsize=16)

    plt.tight_layout()
    plt.savefig("evolution_algorithm.pdf", bbox_inches="tight",pad_inches=0.02)
    plt.show()

#
# if second_figure:
#     U1_MIN = [0.13,0.31]
#     U2_MIN = [0.35,0.5]
#     U_SIM_SIZE = 100
#     U1_MAX=[0.19,0.58]
#     U2_MAX=[0.41,0.62]
#     # ============================
#     # 1. Load JSON
#     # ============================
#     with open("u_sequence.json", "r") as f:
#         u_sequence = json.load(f)
#
#     u_sequence = np.array(u_sequence, dtype=float)
#
#     print("u_sequence shape =", u_sequence.shape)
#     # expected: (n_iter, 4)
#
#     # ============================
#     # 2. Split per well
#     # ============================
#     u1_p1 = u_sequence[:, 0]
#     u2_p1 = u_sequence[:, 1]
#
#     u1_p2 = u_sequence[:, 2]
#     u2_p2 = u_sequence[:, 3]
#
#     iters = np.arange(len(u_sequence))
#
#     # ============================
#     # 3. Figure
#     # ============================
#
#     plt.rcParams.update({
#         "font.size": 14,
#         "axes.titlesize": 18,
#         "axes.labelsize": 16,
#         "xtick.labelsize": 14,
#         "ytick.labelsize": 14,
#     })
#     wells = get_wells()
#     sweeps = []
#     zmin = np.inf
#     zmax = -np.inf
#     i=-1
#     for well, params in wells.items():
#         i=i+1
#         well_name = well
#         well_list = params
#
#         model_NN = build_casadi_surrogate_u2z_for_well(well_name)
#
#         names = [
#             "P_bh_bar",  # 0
#             "P_tb_b_bar",  # 1
#             "w_G_inj",  # 2
#             "w_res",  # 3
#             "w_L_res",  # 4
#             "w_G_res",  # 5
#             "w_w_out",  # 6
#             "w_o_out"  # 7
#         ]
#
#         j = 7
#
#         U1_NN, U2_NN, Z_NN = run_sweep_PINN(
#             model_NN,
#             U1_MIN=U1_MIN[i],
#             U2_MIN=U2_MIN[i],
#             U1_MAX=U1_MAX[i],
#             U2_MAX=U2_MAX[i],
#             U_SIM_SIZE=U_SIM_SIZE
#         )
#         Zj = Z_NN[:, :, j]
#
#         sweeps.append((well_name, U1_NN, U2_NN, Zj))
#
#         zmin = min(zmin, np.nanmin(Zj))
#         zmax = max(zmax, np.nanmax(Zj))
#
#     fill_levels = np.linspace(zmin, zmax, 30)  # smooth color
#     line_levels = np.linspace(zmin, zmax, 30)  # few clean lines
#
#     fig, axes = plt.subplots(
#         2, 1,
#         figsize=(6.5, 8.5),
#         sharex=False,
#         sharey=False,
#         constrained_layout=False
#     )
#
#     mappable = None
#     i=-1
#     for ax, (well_name, U1_NN, U2_NN, Zj) in zip(axes, sweeps):
#         i=i+1
#         u1_opt = wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("u")][0]
#         u2_opt = wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("u")][1]
#         z_opt = wells[well_name]["optima"]["surrogate"]["total_unconstrained"][("w_o_out")]
#         ax, cs = plot_contour(
#             U1=U1_NN,
#             U2=U2_NN,
#             Z=Zj,
#             u1_opt=u1_opt,
#             u2_opt=u2_opt,
#             z_opt=z_opt,
#             title=f"$w_{{out,o}}$($u_1$,$u_2$) - Poço {well_name}",
#             zlabel=names[j],
#             fill_levels=fill_levels,
#             line_levels=line_levels,
#             mark_optimum=False,
#             ax=ax,
#             add_colorbar=False,
#             vmin=zmin,
#             vmax=zmax,
#             just_contour=True
#         )
#         mappable = cs
#
#
#
#     # one shared colorbar
#     # -------------------------
#     # Create a dedicated colorbar axis
#     # -------------------------
#     cbar_ax = fig.add_axes([0.82, 0.10, 0.025, 0.8])
#     # [left, bottom, width, height] in figure coordinates
#
#     cbar = fig.colorbar(mappable, cax=cbar_ax)
#
#     cbar.set_label(r"$w_{out,o} (kg/s)$", fontsize=16)
#
#     cbar.locator = ticker.MaxNLocator(nbins=5)
#     cbar.update_ticks()
#     cbar.ax.tick_params(labelsize=14)
#
#     # ----------------------------
#     # P1 trajectory
#     # ----------------------------
#     ax = axes[0]
#     ax.plot(u1_p1, u2_p1, marker="o", linewidth=1.8)
#     for k, (x, y) in enumerate(zip(u1_p1, u2_p1)):
#         ax.annotate(
#             str(k),
#             (x, y),
#             xytext=(5, 5),
#             textcoords="offset points",
#             fontsize=12
#         )
#         ax.set_xlim(U1_MIN[0],U1_MAX[0])
#         ax.set_ylim(U2_MIN[0],U2_MAX[0])
#
#     # ax.set_title("Trajetória das iterações no plano de controle - Poço P1", fontsize=16, pad=12)
#     # ax.set_xlabel(r"$u_1$", fontsize=14)
#     # ax.set_ylabel(r"$u_2$", fontsize=14)
#     # ax.set_aspect("equal", adjustable="box")
#     # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
#
#     # ----------------------------
#     # P2 trajectory
#     # ----------------------------
#     ax = axes[1]
#     ax.plot(u1_p2, u2_p2, marker="o", linewidth=1.8)
#     for k, (x, y) in enumerate(zip(u1_p2, u2_p2)):
#         ax.annotate(
#             str(k),
#             (x, y),
#             xytext=(5, 5),
#             textcoords="offset points",
#             fontsize=12
#         )
#         ax.set_xlim(U1_MIN[1],U1_MAX[1])
#         ax.set_ylim(U2_MIN[1],U2_MAX[1])
#
#     # ax.set_title("Trajetória das iterações no plano de controle - Poço P2", fontsize=16, pad=12)
#     # ax.set_xlabel(r"$u_1$", fontsize=14)
#     # ax.set_ylabel(r"$u_2$", fontsize=14)
#     # ax.set_aspect("equal", adjustable="box")
#     # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
#
#     # ============================
#     # # 4. Axis limits and ticks
#     # # ============================
#     # ticks = [0.0, 0.25, 0.50, 0.75, 1.00]
#     # for ax in axes:
#     #     ax.set_xlim(0.0, 1.0)
#     #     ax.set_ylim(0.0, 1.0)
#     #     ax.set_xticks(ticks)
#     #     ax.set_yticks(ticks)
#     #     ax.tick_params(labelsize=12)
#
#
#
#
#
#






    #
    # plt.savefig("unconstrained_colormap.pdf", bbox_inches="tight",pad_inches=0.02)
    # plt.show()

from matplotlib.lines import Line2D


# legend_elements = [
#     Line2D(
#         [0], [0],
#         marker="*",
#         color="w",
#         label="Ótimo irrestrito",
#         markerfacecolor="blue",
#         markersize=12
#     )
# ]
#
# fig.legend(
#     handles=legend_elements,
#     loc="lower center",
#     bbox_to_anchor=(-0.20, -0.02),
#     ncol=1,
#     frameon=False,
#     fancybox=True,
#     fontsize=18
# )


#
# if surrogate_wells:
#     for well, params in wells.items():
#         well_name = well
#         well_list = params
#
#         model_sur = make_model(
#             "surrogate",
#             BSW=well_list["BSW"],
#             GOR=well_list["GOR"],
#             PI=well_list["PI"],
#             K_gs=well_list["K_gs_sur"],
#             K_inj=well_list["K_inj_sur"],
#             K_pr=well_list["K_pr_sur"],
#         )
#
#         results = run_sweep(
#             model_sur,
#             U1_MIN=U1_MIN,
#             U2_MIN=U2_MIN,
#             U_SIM_SIZE=U_SIM_SIZE,
#             y_guess_init=np.array(well_list["y_guess_sur"], dtype=float),
#             z_guess_init=None,
#         )
#
#         U1_SUR = np.asarray(results["U1"], dtype=float)
#         U2_SUR = np.asarray(results["U2"], dtype=float)
#
#         W_o_out = np.asarray(results["OUT"]["w_o_out"], dtype=float)
#         P_bh = np.asarray(results["OUT"]["P_bh_bar"], dtype=float)
#         P_tb_b = np.asarray(results["OUT"]["P_tb_b_bar"], dtype=float)
#         w_inj=np.asarray(results["OUT"]["w_G_inj"], dtype=float)
#
#         ax = plot_contour(
#             U1=U1_SUR,
#             U2=U2_SUR,
#             Z=W_o_out,
#             title=f"{well_name}: w_o_out(u1,u2)",
#             zlabel="w_o_out",
#             levels=40,
#         )
#
#         ax = plot_contour(
#             U1=U1_SUR,
#             U2=U2_SUR,
#             Z=P_bh,
#             title=f"{well_name}: P_bh(u1,u2)",
#             zlabel="w_o_out",
#             levels=40,
#         )
#
#
#         ax = plot_contour(
#             U1=U1_SUR,
#             U2=U2_SUR,
#             Z=w_inj,
#             title=f"{well_name}: w_inj(u1,u2)",
#             zlabel="w_inj",
#             levels=40,
#         )
#
#         # overlay_common_boundaries(
#         #     ax=ax,
#         #     U1=U1_SUR,
#         #     U2=U2_SUR,
#         #     P_bh=P_bh,
#         #     P_tb_b=P_tb_b,
#         #     coeff_stability=well_list["coeff_stability"],
#         #     u1_min=U1_MIN,
#         #     u1_max=1.0,
#         #     pbh_min=90.0,
#         #     ptb_max=120.0,
#         #     deg=2,
#         # )
#
#         ax.set_xlim(U1_MIN, 1.0001)
#         ax.set_ylim(U2_MIN, 1.0001)
#         ax.figure.tight_layout()
#         plt.show()
#
#


#
#
#
#
#
#
# if pinn_wells:
#     for well, params in wells.items():
#         well_name=well
#         well_list =params
#
#         # 1. Build actual CasADi PINN-Alg model
#         model_NN=build_casadi_surrogate_u2z_for_well(well_name)
#
#         names = ["P_bh_bar",  # 0
#                  "P_tb_b_bar",  # 1
#                  "w_G_inj",  # 2
#                  "w_res",  # 3
#                  "w_L_res",  # 4
#                  "w_G_res",  # 5
#                  "w_w_out",  # 6
#                  "w_o_out"]  # 7
#
#         j = 7
#
#         U1_NN, U2_NN, Z_NN = run_sweep_PINN(
#             model_NN,
#             U1_MIN=U1_MIN,
#             U2_MIN=U2_MIN,
#             U_SIM_SIZE=U_SIM_SIZE
#         )
#
#         ax=plot_contour(
#             U1=U1_NN,
#             U2=U2_NN,
#             Z=Z_NN[:, :, j],
#             title=f"{names[j]}(u1,u2)",
#             zlabel=names[j],
#             levels=40,
#         )
#
#         coeff = well_list["coeff_stability"]
#         b_hat = np.poly1d(coeff)
#         overlay_boundary_and_fit(ax,b_hat)
#         ax.set_xlim(0.05, 1.0)
#         ax.set_ylim(0.10, 1.0)
#
#         # BH PRESSURE
#         P_bh = Z_NN[:, :, 0]
#         u1_grid = U1_NN[:, 0]
#         u2_grid = U2_NN[0, :]
#         boundary_u1, boundary_u2 = extract_threshold_boundary_from_grid(
#             u1_grid=u1_grid,
#             u2_grid=u2_grid,
#             Z=P_bh,
#             threshold=90.0,
#             mode=">=",
#             side="last_true"
#         )
#         b_hat_pbh = fit_boundary_polynomial(boundary_u1, boundary_u2, deg=2)
#         overlay_boundary_curve(
#             ax,
#             b_hat_pbh,
#             u1_min=0.05,
#             u1_max=1.0,
#             label=r"$P_{bh}=90$ bar",
#             color="red"
#         )
#
#
#         # TUBING PRESSURE
#         P_bh = Z_NN[:, :, 1]
#         u1_grid = U1_NN[:, 1]
#         u2_grid = U2_NN[1, :]
#         boundary_u1, boundary_u2 = extract_threshold_boundary_from_grid(
#             u1_grid=u1_grid,
#             u2_grid=u2_grid,
#             Z=P_bh,
#             threshold=120.0,
#             mode="<=",
#             side="last_true"
#         )
#         b_hat_ttb = fit_boundary_polynomial(boundary_u1, boundary_u2, deg=2)
#         overlay_boundary_curve(
#             ax,
#             b_hat_ttb,
#             u1_min=0.05,
#             u1_max=1.0,
#             label=r"$P_{tb}=120$ bar",
#             color="blue"
#         )
#
#         ax.set_xlim(U1_MIN,1.0001)
#         ax.set_ylim(U2_MIN, 1.0001)
#         ax.figure.tight_layout()
#         ax.figure.show()
#
#
# ###############################
#
# if surrogate_wells:
#
#     # 2. Build symbolic surrogate model
#     model_sur=make_model(
#         "surrogate",
#         BSW=well_list["BSW"],
#         GOR=well_list["GOR"],
#         PI=well_list["PI"],
#         K_gs=well_list["K_gs_sur"],
#         K_inj=well_list["K_inj_sur"],
#         K_pr=well_list["K_pr_sur"],
#     )
#
#     # 3. Sweep ranges
#
#     y_guess_sur=well_list["y_guess_sur"]
#
#
#     # 4. Sweep symbolic surrogate
#     results_sur= run_sweep(model_sur,
#                     U1_MIN=U1_MIN,
#                     U2_MIN=U2_MIN,
#                     U_SIM_SIZE=U_SIM_SIZE,
#                     y_guess_init=y_guess_sur,
#                     z_guess_init=None,
#                     RES_TOL_DX=RES_TOL_DX,
#                     TOL_EIG=TOL_EIG)
#
#     U1_sur = np.asarray(results_sur["U1"], dtype=float)
#     U2_sur = np.asarray(results_sur["U2"], dtype=float)
#
