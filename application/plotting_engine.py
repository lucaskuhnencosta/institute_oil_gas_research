import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl
import numpy as np
from sipbuild.generator.outputs.formatters import value_list

from application.simulation_engine import *

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
    ax.view_init(elev=25, azim=45)

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
                       title,
                       ax=None):
    ######################################################################################
    color_primary = (54 / 255, 32 / 255, 229 / 255)  # blue
    color_secondary = (240 / 255, 101 / 255, 74 / 255)  # orange
    color_third = (183 / 255, 53 / 255, 192 / 255)  # purple (saved)
    #####################################################################
    """
        U1, U2: meshgrid arrays (same shape)
        STABLE: same shape, with values:
            1.0  -> stable
            0.0  -> unstable
            NaN  -> failure/unknown
        """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))


    xlim=(U1_MIN-0.05,1.05),
    ylim=(U2_MIN-0.05,1.05),

    U1=np.asarray(U1,dtype=float)
    U2=np.asarray(U2,dtype=float)

    STABLE=np.asarray(STABLE,dtype=float)

    # fig,ax=plt.subplots(figsize=(8, 8))

    stable_mask=(STABLE==1.0)
    unstab_mask=(STABLE==0.0)


    if np.any(stable_mask):
        ax.scatter(U1[stable_mask],
                   U2[stable_mask],
                   marker='o',
                   s=80,
                   facecolors="none",
                   edgecolors=color_primary,
                   linewidths=1.5,
                   label="Equilíbrio estável")
    if np.any(unstab_mask):
        ax.scatter(U1[unstab_mask],
                   U2[unstab_mask],
                   marker='x',
                   s=80,
                   c=color_secondary,
                   linewidths=1.5,
                   label="Equilíbrio instável"
                   )
    ax.set_xlabel("$u_1$",fontsize=16)
    ax.set_ylabel("$u_2$",fontsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.35)
    ax.set_title(title,fontsize=18)

    ax.tick_params(axis='both', labelsize=14)

    # legend only if there are labeled items
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        leg = ax.legend(
            loc="upper right",
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
    # ax.set_aspect('equal', adjustable='box')
    return ax

def overlay_boundary_and_fit(ax,
                             b_hat):

    u1_dense=np.linspace(0.0,1.01,1000)
    u2_fit=b_hat(u1_dense)

    eq_str=poly_to_string(b_hat,var="u_1")

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
            loc="upper center",
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

def poly_to_string(p, var="u_1", precision=3):
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
            terms.append(f"{c_str}{var}^{{{power}}}")

    return " + ".join(terms).replace("+ -", "- ")
################################################################


################################################################
########################  HEATMAP  #############################

############# We start with a generic plot_contour ##############
import numpy as np
import matplotlib.pyplot as plt


def plot_contour(
    U1,
    U2,
    Z,
    title,
    zlabel="Z",
    u1_opt=None,
    u2_opt=None,
    z_opt=None,
    fill_levels=10,
    line_levels=10,
    mask=None,
    mark_optimum=True,
    ax=None,
    add_colorbar=True,
    vmin=None,
    vmax=None,
    just_contour=False,
    cmap="viridis",
    contour_color="k",
    linewidths=2.0,
    alpha=0.5,
    equal_aspect=False,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
):
    """
    Generic contour plot utility.

    Returns
    -------
    ax : matplotlib axis
    mappable : contourf object if available, otherwise contour object
    """

    Z_plot = np.array(Z, dtype=float, copy=True)

    if mask is not None:
        Z_plot[np.asarray(mask, dtype=bool)] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Filled contour
    cf = None
    if not just_contour:
        cf = ax.contourf(
            U1,
            U2,
            Z_plot,
            levels=fill_levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    # Line contour
    cs = ax.contour(
        U1,
        U2,
        Z_plot,
        levels=line_levels,
        cmap=cmap if just_contour else None,
        colors=contour_color if not just_contour else None,
        linewidths=linewidths,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar
    if add_colorbar:
        mappable = cf if cf is not None else cs
        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_label(zlabel, fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    # Optimum marker
    if mark_optimum and (u1_opt is not None) and (u2_opt is not None):
        if z_opt is None:
            label = "Optimum"
        else:
            label = f"Optimum {z_opt:.3f}"

        ax.plot(
            u1_opt,
            u2_opt,
            "b*",
            markersize=12,
            label=label
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc="best",
                frameon=True,
                fancybox=True,
                fontsize=12
            )
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_alpha(1.0)
            frame.set_linewidth(1.0)

    # Labels and title
    ax.set_xlabel(r"$u_1$", fontsize=16)
    ax.set_ylabel(r"$u_2$", fontsize=16)
    ax.set_title(title, fontsize=18, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Optional equal aspect
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Ticks
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:.2f}" for t in xticks])

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:.2f}" for t in yticks])

    # Return the right mappable for external colorbar use
    mappable = cf if cf is not None else cs
    return ax, mappable
# def plot_contour(
#         U1,
#         U2,
#         Z,
#         u1_opt,
#         u2_opt,
#         z_opt,
#         title,
#         zlabel='Z',
#         fill_levels=40,
#         line_levels=40,
#         mask=None,
#         mark_optimum=True,
#         optimum="max",
#         ax=None,
#         add_colorbar=True,
#         vmin=None,
#         vmax=None,
#         just_contour=False):
#
#     Z_plot=np.array(Z,dtype=float,copy=True)
#
#     if mask is not None:
#         Z_plot[np.asarray(mask,dtype=bool)] = np.nan
#
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))
#     else:
#         fig=ax.figure
#
#     if not just_contour:
#         cf = ax.contourf(U1,
#                          U2,
#                          Z_plot,
#                          levels=fill_levels,
#                          vmin=vmin,
#                          vmax=vmax)
#
#     cs = ax.contour(U1,
#                     U2,
#                     Z_plot,
#                     levels=line_levels,
#                     cmap="viridis",
#                      linewidths=2.0,
#                     alpha=0.9,
#                     vmin=vmin,
#                     vmax=vmax)
#
#     # ax.clabel(cs, inline=True, fontsize=24)
#
#     if add_colorbar:
#         cbar=fig.colorbar(cf, ax=ax, label=zlabel)
#         cbar.set_label(zlabel, fontsize=16)
#
#         # tick labels
#         cbar.ax.tick_params(labelsize=14)
#
#     if mark_optimum:
#         ax.plot(
#             u1_opt,
#             u2_opt,
#             "b*",
#             markersize=14,
#             label=f"Ótimo irrestrito {z_opt:.2f}"
#         )
#
#         handles, labels = ax.get_legend_handles_labels()
#         if handles:
#             leg = ax.legend(
#                 loc="lower center",
#                 frameon=True,
#                 fancybox=True,
#                 fontsize=14
#             )
#             frame = leg.get_frame()
#             frame.set_facecolor("white")
#             frame.set_edgecolor("black")
#             frame.set_alpha(1)
#             frame.set_linewidth(1.0)
#
#     ax.set_xlabel(f"$u_1$",fontsize=16)
#     ax.set_ylabel(f"$u_2$",fontsize=16)
#     ax.set_title(title, fontsize=18,pad=12)
#     ax.tick_params(axis="both", which="major", labelsize=14)
#     ax.set_aspect('equal', adjustable='box')
#
#     ticks = [0.25, 0.5, 0.75, 1.0]
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)
#     ax.set_xlim(0.10,1.00)
#     ax.set_ylim(0.10,1.00)
#     # optional: nicer formatting
#     ax.set_xticklabels([f"{t:.2f}" for t in ticks])
#     ax.set_yticklabels([f"{t:.2f}" for t in ticks])
#     return ax,cs

############# We make a wraper for simulation models ##############

def plot_surrogate_contour(
    results,
    key="w_o_out",
    title="Contour plot",
    zlabel=None,
    only_stable=True,
    only_success=True,
    levels=40,
    mark_optimum=True,
    optimum="max",
):
    U1 = np.asarray(results["U1"], dtype=float)
    U2 = np.asarray(results["U2"], dtype=float)
    Z = np.asarray(results["OUT"][key], dtype=float)

    mask = np.zeros_like(Z, dtype=bool)

    if only_success:
        SUCCESS = np.asarray(results["SUCCESS"], dtype=bool)
        mask |= ~SUCCESS

    if only_stable:
        STABLE = np.asarray(results["STABLE"], dtype=float)
        mask |= ~(STABLE == 1.0)

    if zlabel is None:
        zlabel = key

    return plot_contour(
        U1=U1,
        U2=U2,
        Z=Z,
        title=title,
        zlabel=zlabel,
        levels=levels,
        mask=mask,
        mark_optimum=mark_optimum,
        optimum=optimum,
    )

def overlay_common_boundaries(
        ax,
        U1,
        U2,
        P_bh,
        P_tb_b,
        coeff_stability,
        u1_min=0.05,
        u1_max=1.0,
        pbh_min=90.0,
        ptb_max=120.0,
        deg=2,
):
    # Stability boundary
    b_hat_stab = np.poly1d(coeff_stability)
    overlay_boundary_curve(
        ax,
        b_hat_stab,
        u1_min=u1_min,
        u1_max=u1_max,
        label="stability",
        color="black"
    )

    u1_grid = U1[:, 0]
    u2_grid = U2[0, :]

    # P_bh boundary
    boundary_u1, boundary_u2 = extract_threshold_boundary_from_grid(
        u1_grid=u1_grid,
        u2_grid=u2_grid,
        Z=P_bh,
        threshold=pbh_min,
        mode=">=",
        side="last_true"
    )
    if boundary_u1.size >= deg + 1:
        b_hat_pbh = fit_boundary_polynomial(boundary_u1, boundary_u2, deg=deg)
        overlay_boundary_curve(
            ax,
            b_hat_pbh,
            u1_min=u1_min,
            u1_max=u1_max,
            label=rf"$P_{{bh}}={pbh_min:g}$ bar",
            color="red"
        )

    # P_tb_b boundary
    boundary_u1, boundary_u2 = extract_threshold_boundary_from_grid(
        u1_grid=u1_grid,
        u2_grid=u2_grid,
        Z=P_tb_b,
        threshold=ptb_max,
        mode="<=",
        side="last_true"
    )
    if boundary_u1.size >= deg + 1:
        b_hat_ptb = fit_boundary_polynomial(boundary_u1, boundary_u2, deg=deg)
        overlay_boundary_curve(
            ax,
            b_hat_ptb,
            u1_min=u1_min,
            u1_max=u1_max,
            label=rf"$P_{{tb}}={ptb_max:g}$ bar",
            color="blue"
        )

    return ax

def overlay_boundary_curve(
    ax,
    b_hat,
    u1_min=0.05,
    u1_max=1.0,
    label=None,
    color="k",
    linewidth=2.2,
    linestyle="-",
):
    u1_dense = np.linspace(u1_min, u1_max, 1000)
    u2_fit = b_hat(u1_dense)

    # optional clipping to visible region
    y0, y1 = ax.get_ylim()
    mask = (u2_fit >= y0) & (u2_fit <= y1)

    ax.plot(
        u1_dense[mask],
        u2_fit[mask],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )

    if label is not None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc="best",
                frameon=True,
                fancybox=False,
                fontsize=12
            )
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")
            frame.set_alpha(1)
            frame.set_linewidth(1.0)

    return ax

    #
    # u1_grid = results["u1_grid"]
    # u2_grid = results["u2_grid"]
    # U1=results["U1"]
    # U2=results["U2"]
    #
    #
    # W = np.array(results["OUT"]["w_o_out"], dtype=float)
    # SUCCESS = np.array(results["SUCCESS"], dtype=bool)
    # STABLE = np.array(results["STABLE"], dtype=float)
    #
    # fig,ax=plt.subplots(figsize=(8, 8))
    #
    # mask = np.zeros_like(W, dtype=bool)
    #
    # if only_success:
    #     mask |= ~SUCCESS
    #
    # if only_stable:
    #     mask |= ~(STABLE == 1.0)
    #
    # W_plot = np.array(W, copy=True)
    # W_plot[mask] = np.nan
    #
    # # # Mesh for contour: X must match columns (u2), Y rows (u1)
    # # U2, U1 = np.meshgrid(u2_grid, u1_grid)
    #
    # # plt.figure(figsize=(8, 8))
    # cf = plt.contourf(U1, U2, W_plot, levels=levels)
    # cs = plt.contour(U1, U2, W_plot, levels=levels, linewidths=0.5)
    # ax.clabel(cs, inline=True, fontsize=8)
    #
    # cbar=fig.colorbar(cf,ax=ax, label="w_o_out")
    #
    # # Mark optimum among valid points
    # if np.all(np.isnan(W_plot)):
    #     print("No valid points available to locate optimum.")
    # else:
    #     flat_idx = np.nanargmax(W_plot)
    #     i_opt, j_opt = np.unravel_index(flat_idx, W_plot.shape)
    #
    #
    #     u1_opt = u1_grid[i_opt]
    #     u2_opt = u2_grid[j_opt]
    #     w_opt = W_plot[i_opt, j_opt]
    #
    #
    #     ax.plot(u1_opt,
    #             u2_opt,
    #             'b*',
    #             markersize=14,
    #             label=f"Ótimo {w_opt:.3f}")
    #
    #     handles, labels = ax.get_legend_handles_labels()
    #     if handles:
    #         leg = ax.legend(loc="best",
    #                         frameon=True,
    #                         fancybox=False,
    #                         fontsize=14)
    #         frame = leg.get_frame()
    #         frame.set_facecolor("white")
    #         frame.set_edgecolor("black")
    #         frame.set_alpha(1)
    #         frame.set_linewidth(1.0)
    #
    # ax.set_xlabel("u1")
    # ax.set_ylabel("u2")
    # ax.set_title(title, fontsize=16)
    #
    # return ax
    #     # plt.plot(u2_opt, u1_opt, 'r*', markersize=14, label=f"Optimum = {w_opt:.3f}")
    #     # plt.legend()
    #     #
    #     # print(f"Optimum at u1={u1_opt:.4f}, u2={u2_opt:.4f}, w_o_out={w_opt:.6f}")
    #
    # plt.xlabel("u2")
    # plt.ylabel("u1")
    # plt.title(title,fontsize=16)
    # plt.tight_layout()
    # plt.show()

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

