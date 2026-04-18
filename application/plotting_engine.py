import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl
import numpy as np

################################################################
################### SOME RULES FOR PLOTTING ####################

# TITLE IS ALWAYS BROUGHT TO HERE FROM THE CLIENT, FONTSIZE IS ALWAYS 20
# FIGSIZE IS ALWAYS (8,8) FOR THOSE 2D PLOTS
# SIZE FOR TICKS, LEGENDS AND AXIS TITLES IS ALWAYS 16

################################################################
################### PLOTTING VARS ##############################

PLOT_VARS = [
    # =================================
    # States (kg)
    # =================================
    ("m_G_an", "m_G_an(u1,u2)", "kg"),
    ("m_G_t", "m_G_t(u1,u2)", "kg"),
    ("m_o_t", "m_o_t(u1,u2)", "kg"),
    ("m_w_t", "m_w_t(u1,u2)", "kg"),
    ("m_G_b", "m_G_b(u1,u2)", "kg"),
    ("m_o_b", "m_o_b(u1,u2)", "kg"),
    ("m_w_b", "m_w_b(u1,u2)", "kg"),
    #
    # # =================================
    # # Hold-ups / Volumes (m³)
    # # =================================
    ("V_L_tb_states", "V_L_tb_states(u1,u2)", "m³"),
    ("V_L_bh_states", "V_L_bh_states(u1,u2)", "m³"),
    #
    # # =================================
    # # Volume Fractions (-)
    # # =================================
    ("alpha_L_tb", "alpha_L_tb(u1,u2)", "-"),
    ("alpha_G_tb", "alpha_G_tb(u1,u2)", "-"),
    ("alpha_L_tb_t", "alpha_L_tb_t(u1,u2)", "-"),
    ("alpha_L_tb_b", "alpha_L_tb_b(u1,u2)", "-"),
    ("alpha_L_bh", "alpha_L_bh(u1,u2)", "-"),
    ("alpha_G_bh", "alpha_G_bh(u1,u2)", "-"),
    #
    # # =================================
    # # Pressures (bar)
    # # =================================
    ("P_an_t_bar", "P_an_t_bar(u1,u2)", "bar"),
    ("P_an_bar", "P_an_bar(u1,u2)", "bar"),
    ("P_bh_bar", "P_bh_bar(u1,u2)", "bar"),
    ("P_tb_t_bar", "P_tb_t_bar(u1,u2)", "bar"),
    ("P_tb_b_bar", "P_tb_b_bar(u1,u2)", "bar"),
    ("P_hidro_tb_bar", "P_hidro_tb_bar(u1,u2)", "bar"),
    ("P_hidro_bh_bar", "P_hidro_bh_bar(u1,u2)", "bar"),
    #
    # # =================================
    # # Friction Losses (bar)
    # # =================================
    ("F_t_bar", "Comparison between rigorous and surrogate models for the friction at the tubing", "bar"),
    ("F_bh_bar", "F_bh_bar(u1,u2)", "bar"),
    #
    # # =================================
    # # Delta Pressures (bar)
    # # =================================
    ("dP_int_bar", "dP_int_bar(u1,u2)", "bar"),
    ("dP_gs_an_bar", "dP_gs_an_bar(u1,u2)", "bar"),
    ("dP_an_tb_bar", "dP_an_tb_bar(u1,u2)", "bar"),
    ("dP_res_bh_bar", "dP_res_bh_bar(u1,u2)", "bar"),
    ("dP_tb_choke_bar", "dP_tb_choke_bar(u1,u2)", "bar"),
    #
    # # =================================
    # # Hydrodynamics
    # # =================================
    ("Re_tb", "Re_tb(u1,u2)", "-"),
    ("Re_bh", "Re_bh(u1,u2)", "-"),
    ("U_avg_mix_tb", "U_avg_mix_tb(u1,u2)", "m/s"),
    ("U_avg_b", "U_avg_b(u1,u2)", "m/s"),
    #
    # # =================================
    # # Flow Rates (kg/s)
    # # =================================
    ("w_res", "w_res(u1,u2)", "kg/s"),
    ("w_L_res", "w_L_res(u1,u2)", "kg/s"),
    ("w_G_res", "w_G_res(u1,u2)", "kg/s"),

    ("w_out", "w_out(u1,u2)", "kg/s"),
    ("w_G_out", "w_G_out(u1,u2)", "kg/s"),
    ("w_L_out", "w_L_out(u1,u2)", "kg/s"),
    ("w_w_out", "w_w_out(u1,u2)", "kg/s"),
    ("w_o_out", "w_o_out(u1,u2)", "kg/s"),
    #
    ("w_G_inj", "w_G_inj(u1,u2)", "kg/s"),
    ("w_up", "w_up(u1,u2)", "kg/s"),
    #
    # # =================================
    # # Mixture Properties
    # # =================================
    ("rho_avg_mix_tb", "rho_avg_mix_tb(u1,u2)", "kg/m³"),
]

######################################################################
################### STANDARD SURFACE RESPONSE PLOT HELPER ############
######### IMPORTANT FOR CLEAN SIMULATION RESPONSE FUNCTIONS! #########

def plot_surface(fig,
                 ax,
                 U1,
                 U2,
                 Z,
                 title,
                 zlabel,
                 cmap_name="viridis"):
    Zm = np.ma.masked_invalid(Z)

    if np.all(Zm.mask):
        ax.set_title(title + " (no data)")
        return ax

    zmin = float(Zm.min())
    zmax = float(Zm.max())
    if abs(zmax - zmin) < 1e-12:
        zmax = zmin + 1.0

    norm = Normalize(vmin=zmin, vmax=zmax)
    cmap = mpl.colormaps[cmap_name]

    surf = ax.plot_surface(
        U2, U1, Zm,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        antialiased=False,
        shade=False
    )

    # lighting effect (MATLAB-like shine)
    ax.view_init(elev=25, azim=135)

    cb = fig.colorbar(surf, ax=ax, pad=0.02, shrink=0.75)
    cb.set_label(zlabel)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("$u_2$",fontsize=12)
    ax.set_ylabel("$u_1$",fontsize=12)
    ax.set_zlabel(zlabel,fontsize=12)
    ax.set_xlim(float(U2.min()), float(U2.max()))
    ax.set_ylim(float(U1.min()), float(U1.max()))
    ax.set_xticks(np.round(np.linspace(U2.min(), U2.max(), 6), 2))
    ax.set_yticks(np.round(np.linspace(U1.min(), U1.max(), 6), 2))
    ax.tick_params(axis='both', labelsize=12)

    # leg = ax.legend(
    #     loc="best",
    #     frameon=True,
    #     fancybox=False,
    #     fontsize=14
    # )


##########################################################
################### PLOTING ALL CURVES ###################

# def plot_all_curves(
#         U1,
#         U2,
#         results_all
# ):
#     for var, title, zlabel in PLOT_VARS:
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(1, 1, 1, projection="3d")
#         plot_surface(fig, ax, U1, U2, results_all["rigorous"]["OUT"][var], title, zlabel)
#         ax.view_init(elev=25, azim=45)
#         plt.tight_layout()
#         plt.show()
#     plot_figures(results_all["rigorous"], title="rigorous")
#     ax = plot_stability_map(U1,
#                             U2,
#                             STABLE=results_all["rigorous"]["STABLE"],
#                             title="Stability map + fitted stability constraint")
#


################################################################
########################  STABILITY MAP  #######################

def plot_stability_map(U1,
                       U2,
                       STABLE,
                       U1_MIN,
                       U2_MIN,
                       title):

    """
        U1, U2: meshgrid arrays (same shape)
        STABLE: same shape, with values:
            1.0  -> stable
            0.0  -> unstable
            NaN  -> failure/unknown
        """
    xlim=(U1_MIN-0.05,1.05),
    ylim=(U2_MIN-0.05,1.05),

    U1=np.asarray(U1,dtype=float)
    U2=np.asarray(U2,dtype=float)

    STABLE=np.asarray(STABLE,dtype=float)

    fig,ax=plt.subplots(figsize=(8, 8))

    stable_mask=(STABLE==1.0)
    unstab_mask=(STABLE==0.0)


    if np.any(stable_mask):
        ax.scatter(U1[stable_mask],
                   U2[stable_mask],
                   marker='o',
                   s=150,
                   facecolors="none",
                   edgecolors="green",
                   linewidths=1.5,
                   label="Equilíbrio estável")
    if np.any(unstab_mask):
        ax.scatter(U1[unstab_mask],
                   U2[unstab_mask],
                   marker='x',
                   s=150,
                   c="red",
                   linewidths=1.5,
                   label="Equilíbrio instável"
                   )
    ax.set_xlabel("$u_1$",fontsize=16)
    ax.set_ylabel("$u_2$",fontsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.35)
    ax.set_title(title,fontsize=20)

    ax.tick_params(axis='both', labelsize=16)

    # legend only if there are labeled items
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        leg = ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            fontsize=16
        )

        # Force opaque legend
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_alpha(1)
        frame.set_linewidth(1.0)

    return ax

def overlay_boundary_and_fit(ax,
                             b_hat):

    u1_dense=np.linspace(0.0,1.01,1000)
    u2_fit=b_hat(u1_dense)

    eq_str=poly_to_string(b_hat,var="$u_1$")

    ax.plot(u1_dense,
            u2_fit,
            "k-",
            linewidth=2.2,
            label=rf"$u_2={eq_str}$")
    handles, labels = ax.get_legend_handles_labels()

    ax.set_xlabel("$u_1$",fontsize=16)
    ax.set_ylabel("$u_2$",fontsize=16)

    if handles:
        leg = ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            fontsize=14
        )

        # Force opaque legend
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_alpha(1)
        frame.set_linewidth(1.0)

    return ax

def poly_to_string(p, var="u1", precision=3):
    coef = p.c
    deg = len(coef) - 1

    terms = []
    for i, c in enumerate(coef):
        power = deg - i

        if abs(c) < 1e-12:
            continue

        c_str = f"{c:.{precision}f}"

        if power == 0:
            terms.append(f"{c_str}")
        elif power == 1:
            terms.append(f"{c_str}{var}")
        else:
            terms.append(f"{c_str}{var}^{power}")

    return " + ".join(terms).replace("+ -", "- ")
################################################################


################################################################
########################  HEATMAP  #############################

def plot_w_o_out_contour(results,
                         title,
                         only_stable=True,
                         only_success=True,
                         levels=40):
    u1_grid = results["u1_grid"]
    u2_grid = results["u2_grid"]
    U1=results["U1"]
    U2=results["U2"]
    # u1_grid = np.asarray(results["u1_grid"],dtype=float)
    # u2_grid = np.asarray(results["u2_grid"],dtype=float)

    W = np.array(results["OUT"]["w_o_out"], dtype=float)
    SUCCESS = np.array(results["SUCCESS"], dtype=bool)
    STABLE = np.array(results["STABLE"], dtype=float)

    fig,ax=plt.subplots(figsize=(8, 8))

    mask = np.zeros_like(W, dtype=bool)

    if only_success:
        mask |= ~SUCCESS

    if only_stable:
        mask |= ~(STABLE == 1.0)

    W_plot = np.array(W, copy=True)
    W_plot[mask] = np.nan

    # # Mesh for contour: X must match columns (u2), Y rows (u1)
    # U2, U1 = np.meshgrid(u2_grid, u1_grid)

    # plt.figure(figsize=(8, 8))
    cf = plt.contourf(U1, U2, W_plot, levels=levels)
    cs = plt.contour(U1, U2, W_plot, levels=levels, linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8)

    cbar=fig.colorbar(cf,ax=ax, label="w_o_out")

    # Mark optimum among valid points
    if np.all(np.isnan(W_plot)):
        print("No valid points available to locate optimum.")
    else:
        flat_idx = np.nanargmax(W_plot)
        i_opt, j_opt = np.unravel_index(flat_idx, W_plot.shape)


        u1_opt = u1_grid[i_opt]
        u2_opt = u2_grid[j_opt]
        w_opt = W_plot[i_opt, j_opt]


        ax.plot(u1_opt,
                u2_opt,
                'b*',
                markersize=14,
                label=f"Ótimo {w_opt:.3f}")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best",
                            frameon=True,
                            fancybox=False,
                            fontsize=14)
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_alpha(1)
            frame.set_linewidth(1.0)

    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_title(title, fontsize=16)

    return ax
        # plt.plot(u2_opt, u1_opt, 'r*', markersize=14, label=f"Optimum = {w_opt:.3f}")
        # plt.legend()
        #
        # print(f"Optimum at u1={u1_opt:.4f}, u2={u2_opt:.4f}, w_o_out={w_opt:.6f}")

    plt.xlabel("u2")
    plt.ylabel("u1")
    plt.title(title,fontsize=16)
    plt.tight_layout()
    plt.show()

################################################################
################################################################





    # for ax in axes:
    #     ax.view_init(elev=45, azim=135)
    # plt.tight_layout()
    #
    #
    # fig1.savefig(f"gas_lift_surfaces_{well_name}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    # #################################################################
    # fig2 = plt.figure(figsize=(16, 18))
    # fig2.suptitle(f"History behind the bottom tubing pressure for {title}", fontsize=16)
    # axes = [
    #     fig2.add_subplot(2, 2, 1, projection="3d"),
    #     fig2.add_subplot(2, 2, 2, projection="3d"),
    #     fig2.add_subplot(2, 2, 3, projection="3d"),
    #     fig2.add_subplot(2, 2, 4, projection="3d")
    # ]
    # plot_surface(fig2,axes[0], U1, U2, results["OUT"]["P_tb_b_bar"],  "p_tb_b(u1,u2)", "bar")
    # plot_surface(fig2,axes[1], U1, U2, results["OUT"]["P_tb_t_bar"], "p_tb_t(u1,u2)", "bar")
    # plot_surface(fig2,axes[2], U1, U2, results["OUT"]["P_hidro_tb_bar"], "p_hidrostatic(u1,u2) ~ weight of column", "bar")
    # plot_surface(fig2,axes[3], U1, U2, results["OUT"]["F_t_bar"], "F_tb(u1,u2)", "bar")

    #
    # # #################################################################
    # fig3 = plt.figure(figsize=(16, 18))
    # fig3.suptitle(f"History behind the bottom hole pressure for {title}", fontsize=16)
    #
    # axes = [
    #     fig3.add_subplot(2, 2, 1, projection="3d"),
    #     fig3.add_subplot(2, 2, 2, projection="3d"),
    #     fig3.add_subplot(2, 2, 3, projection="3d"),
    #     fig3.add_subplot(2, 2, 4, projection="3d")
    # ]
    #
    # plot_surface(fig3,axes[0], U1, U2, results["OUT"]["P_bh_bar"],  "p_bh(u1,u2)", "bar")
    # plot_surface(fig3,axes[1], U1, U2, results["OUT"]["P_tb_b_bar"], "p_tb_b(u1,u2)", "bar")
    # plot_surface(fig3,axes[2],U1,U2,results["OUT"]["P_hidro_bh_bar"],"p_hidrostatic","bar")
    # plot_surface(fig3,axes[3], U1, U2, results["OUT"]["F_bh_bar"], "F_bh(u1,u2)", "bar")
    # axes[0].view_init(elev=25, azim=45)
    # axes[1].view_init(elev=25,azim=45)# opposite view
    # axes[2].view_init(elev=25,azim=45)# opposite view
    # axes[3].view_init(elev=45,azim=45)# opposite view
    # plt.tight_layout()
    # plt.show()
    #
    # # #################################################################
    #
    # fig4 = plt.figure(figsize=(16, 18))
    # fig4.suptitle(f"Zoom in the friction term for {title}", fontsize=16)
    #
    # axes = [
    #     fig4.add_subplot(2, 2, 1, projection="3d"),
    #     fig4.add_subplot(2, 2, 2, projection="3d"),
    #     fig4.add_subplot(2, 2, 3, projection="3d"),
    #     fig4.add_subplot(2, 2, 4, projection="3d")
    # ]
    #
    # plot_surface(fig4,axes[0], U1, U2, results["OUT"]["F_t_bar"], "F_t(u1,u2)", "bar")
    # plot_surface(fig4,axes[1], U1, U2, results["OUT"]["rho_avg_mix_tb"],  "alpha_avg(u1,u2)", "bar")
    # plot_surface(fig4,axes[2], U1, U2, results["OUT"]["Re_tb"], "Re_tb(u1,u2)", "bar")
    # plot_surface(fig4,axes[3], U1, U2, results["OUT"]["U_avg_mix_tb"], "U_g(u1,u2)", "bar")
    #
    # for ax in axes:
    #     ax.view_init(elev=45, azim=45)    # opposite view
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # #################################################################
    #
    # fig5 = plt.figure(figsize=(16, 18))
    # fig5.suptitle(f"Reservoir Flows for {title}", fontsize=16)
    #
    # axes = [
    #     fig5.add_subplot(2, 2, 1, projection="3d"),
    #     fig5.add_subplot(2, 2, 2, projection="3d"),
    #     fig5.add_subplot(2, 2, 3, projection="3d"),
    #     fig5.add_subplot(2, 2, 4, projection="3d")
    # ]
    #
    # plot_surface(fig5,axes[0], U1, U2, results["OUT"]["w_res"],  "w_res(u1,u2)", "kg/s")
    # plot_surface(fig5,axes[1], U1, U2, results["OUT"]["w_L_res"], "w_l_res(u1,u2)", "kg/s")
    # plot_surface(fig5,axes[2], U1, U2, results["OUT"]["w_G_res"], "w_g_res(u1,u2)", "kg/s")
    # plot_surface(fig5,axes[3], U1, U2, results["OUT"]["w_out"], "w_out(u1,u2)", "kg/s")
    # for ax in axes:
    #     ax.view_init(elev=45, azim=45)  # opposite view
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # # #################################################################
    # #
    # fig6 = plt.figure(figsize=(16, 18))
    # fig6.suptitle(f"Flows for {title}", fontsize=16)
    # axes = [
    #     fig6.add_subplot(2, 2, 1, projection="3d"),
    #     fig6.add_subplot(2, 2, 2, projection="3d"),
    #     fig6.add_subplot(2, 2, 3, projection="3d"),
    #     fig6.add_subplot(2, 2, 4, projection="3d")
    # ]
    # plot_surface(fig6,axes[0], U1, U2, results["OUT"]["w_G_res"],  "w_g_res(u1,u2)", "kg/s")
    # plot_surface(fig6,axes[1], U1, U2, results["OUT"]["w_G_inj"], "w_g_inj(u1,u2)", "kg/s")
    # plot_surface(fig6,axes[2], U1, U2, results["OUT"]["w_L_res"], "w_g_out(u1,u2)", "kg/s")
    # plot_surface(fig6,axes[3], U1, U2, results["OUT"]["w_L_out"], "Q_out(u1,u2)", "kg/s")
    # for ax in axes:
    #     ax.view_init(elev=45, azim=45)   # opposite view
    # plt.tight_layout()
    # plt.show()
    #
    # #
    # # # #################################################################
    # #
    # fig7 = plt.figure(figsize=(16, 18))
    # fig7.suptitle(f"Delta pressure in the chokes for {title}", fontsize=16)
    #
    # axes = [
    #     fig7.add_subplot(2, 2, 1, projection="3d"),
    #     fig7.add_subplot(2, 2, 2, projection="3d"),
    #     fig7.add_subplot(2, 2, 3, projection="3d"),
    #     fig7.add_subplot(2, 2, 4, projection="3d")
    # ]
    #
    # plot_surface(fig7,axes[0], U1, U2, results["OUT"]["dP_gs_an_bar"],  "DP_GS_AN(u1,u2)", "bar")
    # plot_surface(fig7,axes[1], U1, U2,results["OUT"]["dP_an_tb_bar"], "DP_AN_TB(u1,u2)", "bar")
    # plot_surface(fig7,axes[2], U1, U2, results["OUT"]["dP_tb_choke_bar"], "DP_TB_CHOKE(u1,u2)", "bar")
    # plot_surface(fig7,axes[3], U1, U2, results["OUT"]["dP_res_bh_bar"], "DP_RES_BH(u1,u2)", "bar")
    # for ax in axes:
    #     ax.view_init(elev=45, azim=45)   # opposite view
    # plt.tight_layout()
    # plt.show()
    # #
    # # # #################################################################
    # #
    # fig8 = plt.figure(figsize=(16, 18))
    # fig8.suptitle(f"Production for {title}", fontsize=16)
    # axes = [
    #     fig8.add_subplot(2, 2, 1, projection="3d"),
    #     fig8.add_subplot(2, 2, 2, projection="3d"),
    #     fig8.add_subplot(2, 2, 3, projection="3d"),
    #     fig8.add_subplot(2, 2, 4, projection="3d")
    # ]
    # plot_surface(fig8,axes[0], U1, U2, results["OUT"]["w_out"], "Total production", "kg/s")
    # plot_surface(fig8,axes[1], U1, U2, results["OUT"]["w_L_out"], "Production of liquid", "kg/s")
    # plot_surface(fig8,axes[2], U1, U2, results["OUT"]["w_o_out"], "Production of oil", "kg/s")
    # plot_surface(fig8,axes[3], U1, U2, results["OUT"]["w_w_out"], "Production of water`", "kg/s")
    # for ax in axes:
    #     ax.view_init(elev=45, azim=45)   # opposite view
    # plt.tight_layout()
    # plt.show()
    #
    # # #################################################################




    # THIS ONE IS FOR LATER ON
    #
    # def plot_overlay_surface(varname, U1, U2, results_all,
    #                          title="Tubing friction comparison between rigorous and surrogate models", zlabel=""):
    #     """
    #     Plots rigorous (blue) and surrogate (red) surfaces
    #     on the SAME 3D axis.
    #     """
    #     color_u1 = (252 / 255, 77 / 255, 45 / 255)  # rgba(252, 77, 45)
    #     color_u2 = (21 / 255, 59 / 255, 131 / 255)  # rgba(21, 59, 131)
    #
    #     Zr = results_all["rigorous"]["OUT"][varname]
    #     Zs = results_all["surrogate"]["OUT"][varname]
    #
    #     Zr_m = np.ma.masked_invalid(Zr)
    #     Zs_m = np.ma.masked_invalid(Zs)
    #
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection="3d")
    #
    #     ax.plot_surface(
    #         U2, U1, Zr_m,
    #         color=color_u1,
    #         alpha=1.0,
    #         edgecolor="none",
    #         label="Rigorous"
    #     )
    #
    #     ax.plot_surface(
    #         U2, U1, Zs_m,
    #         color=color_u2,
    #         alpha=1.0,
    #         edgecolor="none",
    #         label="Surrogate"
    #     )
    #
    #     ax.set_title(title or varname, fontsize=18)
    #     ax.set_xlabel("u2", fontsize=16)
    #     ax.set_ylabel("u1", fontsize=15)
    #     ax.set_zlabel(zlabel)
    #     ax.set_xlabel("u2", fontsize=16)
    #     ax.set_ylabel("u1", fontsize=16)
    #     ax.set_zlabel(zlabel, fontsize=16)
    #
    #     ax.view_init(elev=30, azim=135)
    #
    #     # Manual legend (since plot_surface doesn't auto-legend)
    #     from matplotlib.lines import Line2D
    #     legend_elements = [
    #         Line2D([0], [0], marker='s', color='w', label='Rigorous',
    #                markerfacecolor=color_u1, markersize=10),
    #         Line2D([0], [0], marker='s', color='w', label='Surrogate',
    #                markerfacecolor=color_u2, markersize=10),
    #     ]
    #     ax.legend(handles=legend_elements, loc="upper right", fontsize=16)
    #
    #     plt.tight_layout()
    #     plt.savefig(f"{varname}.png", dpi=300, bbox_inches="tight")
    #     plt.show()
    #
    #     # plt.savefig(f"{varname}.pdf", bbox_inches="tight")
    #
    #
    #
    #

