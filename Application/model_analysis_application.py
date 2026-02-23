import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from Surrogate_ODE_Model.glc_01_casadi import glc_casadi
from Surrogate_ODE_Model.glc_02_bsw_casadi import glc_bsw_casadi
from Rigorous_DAE_model.glc_rigorous_dae import glc_rigorous_casadi

from Utilities.block_builders import build_steady_state_model
from Solvers.solve_glc_ode_equilibrium import solve_equilibrium_ipopt, model_rigorous
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl

# ---------------------------------------------------------------------
# User Configuration
# ---------------------------------------------------------------------
SIM_KIND="rigorous" # <-- change to "surrogate" or "rigorous"

u1_stable, u2_stable = [], []
u1_unstab, u2_unstab = [], []

u1_grid = np.linspace(0.05, 1.00001, 20)
u2_grid=np.linspace(0.15,1.00001,20)

RES_TOL_DX=1e-6
RES_TOL_G=1e-6
TOL_EIG=1e-8

# Initial guesses (continuation will update them on success)
y_guess_init = np.array([3919.7688, 437.16663, 7956.1206], dtype=float)
z_guess_init = np.array([120e5, 140e5, 10.0], dtype=float)  # only for rigorous DAE: [P_tb, P_bh, w_res]


# ---------------------------------------------------------------------
# Model factory (switch surrogate/rigorous in one place)
# ---------------------------------------------------------------------
def make_model(sim_kind:str):
    sim_kind = sim_kind.lower().strip()
    if sim_kind == "surrogate":
        # ODE: dx,out = glc_casadi(y,u)
        model=build_steady_state_model(
            f_func=glc_casadi,
            state_size=3,
            control_size=2,
            alg_size=None,
            name="glc_surrogate_ss"
        )
        return model
    elif sim_kind == "rigorous":
        # DAE: dx,g,out=glc_rigorous_casadi(y,z,alg_u)
        model=build_steady_state_model(
            f_func=glc_rigorous_casadi,
            state_size=3,
            control_size=2,
            alg_size=3,
            name="glc_rigorous_ss"
        )
        return model
    else:
        raise ValueError("sim_kind must be 'surrogate' or 'rigorous'")

# ---------------------------------------------------------------------
# Robust sweep runner
# ---------------------------------------------------------------------

y0_fixed = np.array([3919.7688, 437.16663, 7956.1206], dtype=float)
z_guess = [120e5, 140e5, 10.0]  # [P_tb, P_bh, w_res] initial guesses (example)
model = build_steady_state_model(glc_rigorous_casadi,
                                 state_size=3,
                                 alg_size=3,
                                 control_size=2,
                                 name="glc")

U1, U2 = np.meshgrid(u1_grid, u2_grid,indexing='ij')
RES_TOL = 1e-6
min_dp_choke_pressure=500

def run_sweep(model,u1_grid,u2_grid,y_guess_init,z_guess_init=None):
    nx=model["nx"]
    nu=model["nu"]
    is_dae=bool(model["is_dae"])

    Z_NAMES=model["Z_names"]
    n_out=len(Z_NAMES)

    Nu1=len(u1_grid)
    Nu2=len(u2_grid)

    OUT = {name: np.full((Nu1, Nu2), np.nan, dtype=float) for name in Z_NAMES}


    RES_DX= np.full((Nu1, Nu2), np.nan, dtype=float)
    RES_G=np.full((Nu1, Nu2), np.nan, dtype=float) if is_dae else None

    STABLE = np.full((Nu1, Nu2), np.nan, dtype=float)  # 1 stable, 0 unstable, NaN unknown/fail
    SUCCESS = np.zeros((Nu1, Nu2), dtype=bool)

    y_guess=np.array(y_guess_init,dtype=float).reshape(-1)
    z_guess=None if (not is_dae) else np.array(z_guess_init,dtype=float).reshape(-1)

    for i, u1 in enumerate(u1_grid):
        for j, u2 in enumerate(u2_grid):
            print("\n----------------------------------")
            print(f"u1={u1} u2={u2}")
            try:
                if is_dae:
                    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                        model=model,
                        u_val=[u1, u2],
                        y_guess=y_guess,
                        z_guess=z_guess
                    )
                else:
                    y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                    model=model,
                    u_val=[u1, u2],
                    y_guess=y_guess
                    )

                print("IPOPT success:", bool(stats.get("success", False)))
                print("IPOPT status :", stats.get("return_status", ""))

                # if not bool(stats.get("success", False)):
                #     continue

                dx_np = np.array(dx_star, dtype=float).reshape(-1)
                res_dx=float(np.linalg.norm(dx_np))
                RES_DX[i,j]=res_dx


                if is_dae:
                    g_np=np.array(g_star, dtype=float).reshape(-1)
                    res_g=float(np.linalg.norm(g_np))
                    RES_G[i,j]=res_g

                    if (res_dx > RES_TOL_DX) or (res_g > RES_TOL_G):
                        print(f"High residuals: ||dx||={res_dx:.3e}, ||g||={res_g:.3e}")
                        continue
                else:
                    if res_dx > RES_TOL_DX:
                        print(f"High residual: ||dx||={res_dx:.3e}")
                        continue

                out_np = np.array(out_star, dtype=float).reshape(-1)
                if out_np.size != n_out:
                    print(f"WARNING: out size {out_np.size} != Z_NAMES {n_out}. Not storing OUT for this point.")
                else:
                    for k, name in enumerate(Z_NAMES):
                        OUT[name][i, j] = out_np[k]

                STABLE[i, j] = 1.0 if stable else 0.0
                SUCCESS[i, j] = True

                if stable:
                    y_guess=np.array(y_star,dtype=float).reshape(-1)
                    if is_dae:
                        z_guess=np.array(z_star,dtype=float).reshape(-1)

                print("Accepted. y*:",y_star,"||dx||:", res_dx, "stable:", stable)
            except Exception as e:
                print("Exception:", repr(e))
                continue
    return {
        "OUT": OUT,
        "RES_DX": RES_DX,
        "RES_G":RES_G,
        "STABLE": STABLE,
        "SUCCESS": SUCCESS,
        "u1_grid": u1_grid,
        "u2_grid": u2_grid,
        "Z_NAMES": Z_NAMES,
        "is_dae": is_dae
    }

def plot_surface(fig, ax, U1, U2, Z, title, zlabel, cmap_name="viridis"):
    Zm = np.ma.masked_invalid(Z)

    if np.all(Zm.mask):
        ax.set_title(title + " (no data)")
        return

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
        antialiased=True,
        shade=True
    )

    # lighting effect (MATLAB-like shine)
    ax.view_init(elev=25, azim=-135)

    cb = fig.colorbar(surf, ax=ax, pad=0.02, shrink=0.75)
    cb.set_label(zlabel)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("u2")
    ax.set_ylabel("u1")
    ax.set_zlabel(zlabel)
    ax.set_xlim(float(U2.min()), float(U2.max()))
    ax.set_ylim(float(U1.min()), float(U1.max()))
    ax.set_xticks(np.round(np.linspace(float(U1.min()), float(U1.max()), 6), 2))
    ax.set_yticks(np.round(np.linspace(float(U2.min()), float(U2.max()), 6), 2))


model = make_model(SIM_KIND)
results = run_sweep(
    model=model,
    u1_grid=u1_grid,
    u2_grid=u2_grid,
    y_guess_init=y_guess_init,
    z_guess_init=z_guess_init
)

#################################################################
# fig1 = plt.figure(figsize=(16, 18))
# fig1.suptitle("MacroPressures surfaces vs Gas Lift Controls", fontsize=16)
#
# axes = [
#     fig1.add_subplot(2, 2, 1, projection="3d"),
#     fig1.add_subplot(2, 2, 2, projection="3d"),
#     fig1.add_subplot(2, 2, 3, projection="3d"),
#     fig1.add_subplot(2, 2, 4, projection="3d")
# ]
# plot_surface(fig1,axes[0], U1, U2, results["OUT"]["P_an_t_bar"], "p_an_t(u1,u2)", "bar")
# plot_surface(fig1,axes[1], U1, U2, results["OUT"]["P_an_b_bar"], "p_an_b(u1,u2)", "bar")
# plot_surface(fig1,axes[2], U1, U2, results["OUT"]["P_tb_b_bar"],  "p_tb_b(u1,u2)", "bar")
# plot_surface(fig1,axes[3], U1, U2, results["OUT"]["P_bh_bar"], "p_bh(u1,u2)", "bar")
# for ax in axes:
#     ax.view_init(elev=45, azim=45)
# plt.tight_layout()
# plt.show()

#################################################################
fig2 = plt.figure(figsize=(16, 18))
fig2.suptitle("History behind the bottom tubing pressure", fontsize=16)
axes = [
    fig2.add_subplot(2, 2, 1, projection="3d"),
    fig2.add_subplot(2, 2, 2, projection="3d"),
    fig2.add_subplot(2, 2, 3, projection="3d"),
    fig2.add_subplot(2, 2, 4, projection="3d")
]
plot_surface(fig2,axes[0], U1, U2, results["OUT"]["P_tb_b_bar"],  "p_tb_b(u1,u2)", "bar")
plot_surface(fig2,axes[1], U1, U2, results["OUT"]["P_tb_t_bar"], "p_tb_t(u1,u2)", "bar")
plot_surface(fig2,axes[2], U1, U2, results["OUT"]["rho_avg_mix_tb"], "p_hidrostatic(u1,u2) ~ weight of column", "bar")
plot_surface(fig2,axes[3], U1, U2, results["OUT"]["F_t_bar"], "F_tb(u1,u2)", "bar")
axes[0].view_init(elev=25, azim=45)
axes[1].view_init(elev=25,azim=45)# opposite view
axes[2].view_init(elev=25,azim=45)# opposite view
axes[3].view_init(elev=45,azim=45)# opposite view
plt.tight_layout()
plt.show()
#
# #################################################################
#
# #################################################################
fig3 = plt.figure(figsize=(16, 18))
fig3.suptitle("History behind the bottom hole pressure", fontsize=16)

axes = [
    fig3.add_subplot(2, 2, 1, projection="3d"),
    fig3.add_subplot(2, 2, 2, projection="3d"),
    fig3.add_subplot(2, 2, 3, projection="3d"),
    fig3.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig2,axes[0], U1, U2, results["OUT"]["P_bh_bar"],  "p_bh(u1,u2)", "bar")
plot_surface(fig2,axes[1], U1, U2, results["OUT"]["P_tb_b_bar"], "p_tb_b(u1,u2)", "bar")
plot_surface(fig3,axes[2],U1,U2,results["OUT"]["rho_avg_"])
plot_surface(fig2,axes[3], U1, U2, F_BH_BAR, "F_bh(u1,u2)", "bar")
axes[0].view_init(elev=25, azim=45)
axes[1].view_init(elev=25,azim=45)# opposite view
axes[2].view_init(elev=25,azim=45)# opposite view
axes[3].view_init(elev=45,azim=45)# opposite view
plt.tight_layout()
plt.show()
#
#
# #################################################################
#
# fig4 = plt.figure(figsize=(16, 18))
# fig4.suptitle("Zoom in the friction term", fontsize=16)
#
# axes = [
#     fig4.add_subplot(2, 2, 1, projection="3d"),
#     fig4.add_subplot(2, 2, 2, projection="3d"),
#     fig4.add_subplot(2, 2, 3, projection="3d"),
#     fig4.add_subplot(2, 2, 4, projection="3d")
# ]
#
# plot_surface(fig4,axes[0], U1, U2, F_T_BAR, "F_t(u1,u2)", "bar")
# plot_surface(fig4,axes[1], U1, U2, ALPHA_AVG_L_TB,  "alpha_avg(u1,u2)", "bar")
# plot_surface(fig4,axes[2], U1, U2, RE_TB, "Re_tb(u1,u2)", "bar")
# plot_surface(fig4,axes[3], U1, U2, U_AVG_G_TB, "U_g(u1,u2)", "bar")
#
# for ax in axes:
#     ax.view_init(elev=45, azim=45)    # opposite view
#
# plt.tight_layout()
# plt.show()
#
#
# #################################################################
#
# fig5 = plt.figure(figsize=(16, 18))
# fig5.suptitle("Reservoir Flows", fontsize=16)
#
# axes = [
#     fig5.add_subplot(2, 2, 1, projection="3d"),
#     fig5.add_subplot(2, 2, 2, projection="3d"),
#     fig5.add_subplot(2, 2, 3, projection="3d"),
#     fig5.add_subplot(2, 2, 4, projection="3d")
# ]
#
# plot_surface(fig5,axes[0], U1, U2, W_RES,  "w_res(u1,u2)", "kg/s")
# plot_surface(fig5,axes[1], U1, U2, W_L_RES, "w_l_res(u1,u2)", "kg/s")
# plot_surface(fig5,axes[2], U1, U2, W_G_RES, "w_g_res(u1,u2)", "kg/s")
# plot_surface(fig5,axes[3], U1, U2, W_OUT, "w_out(u1,u2)", "kg/s")
# for ax in axes:
#     ax.view_init(elev=45, azim=45)  # opposite view
#
# plt.tight_layout()
# plt.show()
#
# #################################################################
#
# fig6 = plt.figure(figsize=(16, 18))
# fig6.suptitle("Outlet Flows", fontsize=16)
#
# axes = [
#     fig6.add_subplot(2, 2, 1, projection="3d"),
#     fig6.add_subplot(2, 2, 2, projection="3d"),
#     fig6.add_subplot(2, 2, 3, projection="3d"),
#     fig6.add_subplot(2, 2, 4, projection="3d")
# ]
#
# plot_surface(fig6,axes[0], U1, U2, W_G_RES,  "w_g_res(u1,u2)", "kg/s")
# plot_surface(fig6,axes[1], U1, U2, W_G_INJ, "w_g_inj(u1,u2)", "kg/s")
# plot_surface(fig6,axes[2], U1, U2, W_G_OUT, "w_g_out(u1,u2)", "kg/s")
# plot_surface(fig6,axes[3], U1, U2, Q_OUT, "Q_out(u1,u2)", "kg/s")
# for ax in axes:
#     ax.view_init(elev=45, azim=45)   # opposite view
#
# plt.tight_layout()
# plt.show()
#
#
# #################################################################
#
# fig7 = plt.figure(figsize=(16, 18))
# fig7.suptitle("Delta pressure in the chokes", fontsize=16)
#
# axes = [
#     fig7.add_subplot(2, 2, 1, projection="3d"),
#     fig7.add_subplot(2, 2, 2, projection="3d"),
#     fig7.add_subplot(2, 2, 3, projection="3d"),
#     fig7.add_subplot(2, 2, 4, projection="3d")
# ]
#
# plot_surface(fig7,axes[0], U1, U2, DP_GS_AN_BAR,  "DP_GS_AN(u1,u2)", "bar")
# plot_surface(fig7,axes[1], U1, U2, DP_AN_TB_BAR, "DP_AN_TB(u1,u2)", "bar")
# plot_surface(fig7,axes[2], U1, U2, DP_TB_CHOKE_BAR, "DP_TB_CHOKE(u1,u2)", "bar")
# plot_surface(fig7,axes[3], U1, U2, DP_RES_BH_BAR, "DP_RES_BH(u1,u2)", "bar")
# for ax in axes:
#     ax.view_init(elev=45, azim=45)   # opposite view
#
# plt.tight_layout()
# plt.show()
#
#
# #################################################################
#
# fig8 = plt.figure(figsize=(16, 18))
# fig8.suptitle("Production", fontsize=16)
#
# axes = [
#     fig8.add_subplot(2, 2, 1, projection="3d"),
#     fig8.add_subplot(2, 2, 2, projection="3d"),
#     fig8.add_subplot(2, 2, 3, projection="3d"),
#     fig8.add_subplot(2, 2, 4, projection="3d")
# ]
#
# plot_surface(fig8,axes[0], U1, U2, W_OUT, "Total production", "kg/s")
# plot_surface(fig8,axes[1], U1, U2, W_L_OUT, "Production of liquid", "kg/s")
# plot_surface(fig8,axes[2], U1, U2, W_L_O_OUT, "Production of oil", "kg/s")
# plot_surface(fig8,axes[3], U1, U2, W_L_W_OUT, "Production of water`", "kg/s")
# for ax in axes:
#     ax.view_init(elev=45, azim=45)   # opposite view
#
# plt.tight_layout()
# plt.show()
#
# plt.figure()
#
# # # Overlay: boundary points + polynomial fit
# # if len(boundary_u1) > 0:
# #     plt.plot(boundary_u1, boundary_u2, "k.", markersize=8, label="boundary pts (min stable u2)")
# #     plt.plot(u1_dense, u2_fit, "k-", linewidth=2, label=f"poly fit deg={deg}")
# if len(u1_stable) > 0:
#     plt.plot(u1_stable, u2_stable, "o", label="stable (O)")
# if len(u1_unstab) > 0:
#     plt.plot(u1_unstab, u2_unstab, "x", label="unstable (X)")
# plt.xlabel("u1")
# plt.ylabel("u2")
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.grid(True)
# # plt.legend()
# plt.title("Stability map in (u1,u2) plane (solver failures not shown)")
# plt.show()


















#############################################
#############################################
#############################################
#############################################
#############################################
#############################################

# # ============================================================
# # Stability constraint you can use later:
# #   u2 >= b_hat(u1) + delta
# # ============================================================
# delta = 0.02  # safety margin in u2 units (tune)
#
# def stability_constraint(u1, u2):
#     return u2 - (float(b_hat(u1)) + delta)   # >= 0 means "predicted stable"
#


# # ============================================================
# # Build an estimated stability boundary u2 = b(u1)
# # from the stable points found in the sweep
# # ============================================================
#
# u1_stable_arr = np.array(u1_stable, dtype=float)
# u2_stable_arr = np.array(u2_stable, dtype=float)
#
# # For each u1 grid value, take the smallest u2 that was stable
# u1_vals = np.unique(u1_stable_arr)
#
# boundary_u1 = []
# boundary_u2 = []
#
# for u1v in u1_vals:
#     mask = np.isclose(u1_stable_arr, u1v)
#     if np.any(mask):
#         boundary_u1.append(u1v)
#         boundary_u2.append(np.min(u2_stable_arr[mask]))
#
# boundary_u1 = np.array(boundary_u1)
# boundary_u2 = np.array(boundary_u2)
#
# # Optional: sort by u1
# idx = np.argsort(boundary_u1)
# boundary_u1 = boundary_u1[idx]
# boundary_u2 = boundary_u2[idx]
#
# print("\nBoundary points (u2 = min stable u2 for each u1):")
# for a, b in zip(boundary_u1, boundary_u2):
#     print(f"u1={a:.3f}  ->  u2*={b:.3f}")

# ============================================================
# Polynomial fit: u2_hat = b_hat(u1)
# ============================================================
#
# deg = 2  # try 2, 3, or 4 (start with 3)
# if len(boundary_u1) < deg + 1:
#     raise RuntimeError(f"Not enough boundary points ({len(boundary_u1)}) to fit degree {deg} polynomial.")
#
# poly_coef = np.polyfit(boundary_u1, boundary_u2, deg=deg)   # highest power first
# b_hat = np.poly1d(poly_coef)
#
# print("\nPolynomial coefficients (highest power first):")
# print(poly_coef)
#
# # Evaluate fitted curve on a dense grid for plotting
# u1_dense = np.linspace(np.min(u1_grid), np.max(u1_grid), 200)
# u2_fit = b_hat(u1_dense)
#
# # Optional: clip to [0,1] to keep it in admissible range
# u2_fit = np.clip(u2_fit, 0.0, 1.0)# ============================================================
# # Polynomial fit: u2_hat = b_hat(u1)
# # ============================================================
#
# from matplotlib.colors import Normalize

#
    # P_AN_T_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # P_AN_B_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # RHO_G_AN_B= np.full((Nu1, Nu2), np.nan, dtype=float)
    # RHO_G_IN= np.full((Nu1, Nu2), np.nan, dtype=float)
    #
    # DP_GS_AN_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # W_G_IN_ORIGINAL= np.full((Nu1, Nu2), np.nan, dtype=float)
    # W_G_IN= np.full((Nu1, Nu2), np.nan, dtype=float)
    #
    # V_GAS_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # V_GAS_TB_T_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # RHO_G_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # P_TB_T_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    #
    # RHO_AVG_MIX_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # ALPHA_AVG_L_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # ALPHA_G_TB_B= np.full((Nu1, Nu2), np.nan, dtype=float)
    # U_AVG_L_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # DENOM_G= np.full((Nu1, Nu2), np.nan, dtype=float)
    # DENOM_G_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # U_AVG_G_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # U_AVG_MIX_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    #
    # RE_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # RE_TB_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # LOG_ARG_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # LOG_ARG_TB_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # LAMBDA_TB= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # F_T_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # P_TB_B_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # DP_AN_TB_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_G_INJ= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # U_AVG_L_BH= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # RE_BH= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # LOG_ARG_BH= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # LAMBDA_BH= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # F_BH_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # P_BH_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # DP_RES_BH_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_RES= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_L_RES= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_G_RES= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # RHO_G_TB_B= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # DENOM_ALPHA_B= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # DENOM_ALPHA_B_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # ALPHA_L_TB_B= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # ALPHA_L_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # RHO_MIX_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # RHO_MIX_TB_T_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # DP_TB_CHOKE_BAR= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # Q_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # DENOM_ALPHA_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # DENOM_ALPHA_T_SAFE= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # ALPHA_G_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_G_TB_T= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_G_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_L_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    # #
    # # W_L_W_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    # # W_L_O_OUT= np.full((Nu1, Nu2), np.nan, dtype=float)
    #
    # P_AN_T_BAR[i, j] = z_np[0]
    # P_AN_B_BAR[i, j] = z_np[1]
    # RHO_G_AN_B[i, j] = z_np[2]
    # RHO_G_IN[i, j] = z_np[3]
    #
    # DP_GS_AN_BAR[i, j] = z_np[4]
    # W_G_IN_ORIGINAL[i, j] = z_np[5]
    # W_G_IN[i, j] = z_np[6]
    #
    # V_GAS_TB_T[i, j] = z_np[7]
    # V_GAS_TB_T_SAFE[i, j] = z_np[8]
    # RHO_G_TB_T[i, j] = z_np[9]
    # P_TB_T_BAR[i, j] = z_np[10]
    #
    # RHO_AVG_MIX_TB[i, j] = z_np[11] * 9.81 * 2048 / 1e5
    # ALPHA_AVG_L_TB[i, j] = z_np[12]
    # ALPHA_G_TB_B[i, j] = z_np[13]
    # U_AVG_L_TB[i, j] = z_np[14]
    # DENOM_G[i, j] = z_np[15]
    # DENOM_G_SAFE[i, j] = z_np[16]
    # U_AVG_G_TB[i, j] = z_np[17]
    # U_AVG_MIX_TB[i, j] = z_np[18]
    #
    # RE_TB[i, j] = z_np[19]
    # RE_TB_SAFE[i, j] = z_np[20]
    # LOG_ARG_TB[i, j] = z_np[21]
    # LOG_ARG_TB_SAFE[i, j] = z_np[22]
    # LAMBDA_TB[i, j] = z_np[23]
    # F_T_BAR[i, j] = z_np[24]
    #
    # P_TB_B_BAR[i, j] = z_np[25]
    # DP_AN_TB_BAR[i, j] = z_np[26]
    # W_G_INJ[i, j] = z_np[27]
    #
    # U_AVG_L_BH[i, j] = z_np[28]
    # RE_BH[i, j] = z_np[29]
    # LOG_ARG_BH[i, j] = z_np[30]
    # LAMBDA_BH[i, j] = z_np[31]
    # F_BH_BAR[i, j] = z_np[32]
    # P_BH_BAR[i, j] = z_np[33]
    #
    # DP_RES_BH_BAR[i, j] = z_np[34]
    # W_RES[i, j] = z_np[35]
    # W_L_RES[i, j] = z_np[36]
    # W_G_RES[i, j] = z_np[37]
    # RHO_G_TB_B[i, j] = z_np[38]
    #
    # DENOM_ALPHA_B[i, j] = z_np[39]
    # DENOM_ALPHA_B_SAFE[i, j] = z_np[40]
    # ALPHA_L_TB_B[i, j] = z_np[41]
    # ALPHA_L_TB_T[i, j] = z_np[42]
    #
    # RHO_MIX_TB_T[i, j] = z_np[43]
    # RHO_MIX_TB_T_SAFE[i, j] = z_np[44]
    #
    # DP_TB_CHOKE_BAR[i, j] = z_np[45]
    # W_OUT[i, j] = z_np[46]
    # Q_OUT[i, j] = z_np[47]
    # DENOM_ALPHA_T[i, j] = z_np[48]
    # DENOM_ALPHA_T_SAFE[i, j] = z_np[49]
    # ALPHA_G_TB_T[i, j] = z_np[50]
    # W_G_OUT[i, j] = z_np[51]
    # W_L_OUT[i, j] = z_np[52]
    #
    # W_L_W_OUT[i, j] = z_np[53]
    # W_L_O_OUT[i, j] = z_np[54]
