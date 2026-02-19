import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from numpy.ma.core import arange
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from Rigorous_ODE_Model.glc_casadi import glc_casadi
from Solvers.solve_glc_ode_equilibrium import solve_equilibrium_ipopt
from Utilities.block_builders import build_steady_state_model
import matplotlib as mpl

"""
This file runs a complete diagnostic on the model.
It needs to run the build_steady_state_model in "block builders" to build a model
It then runs a sweep from different operating points and make plots using continuation
"""


u1_stable, u2_stable = [], []
u1_unstab, u2_unstab = [], []

u_grid = arange(0.05, 1.00, 0.05)
U1, U2 = np.meshgrid(u_grid, u_grid)   # default indexing="xy"
Nu = len(u_grid)

P_AN_T_BAR= np.full((Nu, Nu), np.nan, dtype=float)
P_AN_B_BAR= np.full((Nu, Nu), np.nan, dtype=float)
RHO_G_AN_B= np.full((Nu, Nu), np.nan, dtype=float)
RHO_G_IN= np.full((Nu, Nu), np.nan, dtype=float)

DP_GS_AN_BAR= np.full((Nu, Nu), np.nan, dtype=float)
W_G_IN_ORIGINAL= np.full((Nu, Nu), np.nan, dtype=float)
W_G_IN= np.full((Nu, Nu), np.nan, dtype=float)

V_GAS_TB_T= np.full((Nu, Nu), np.nan, dtype=float)
V_GAS_TB_T_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
RHO_G_TB_T= np.full((Nu, Nu), np.nan, dtype=float)
P_TB_T_BAR= np.full((Nu, Nu), np.nan, dtype=float)

RHO_AVG_MIX_TB= np.full((Nu, Nu), np.nan, dtype=float)
ALPHA_AVG_L_TB= np.full((Nu, Nu), np.nan, dtype=float)
ALPHA_G_TB_B= np.full((Nu, Nu), np.nan, dtype=float)
U_AVG_L_TB= np.full((Nu, Nu), np.nan, dtype=float)
DENOM_G= np.full((Nu, Nu), np.nan, dtype=float)
DENOM_G_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
U_AVG_G_TB= np.full((Nu, Nu), np.nan, dtype=float)
U_AVG_MIX_TB= np.full((Nu, Nu), np.nan, dtype=float)

RE_TB= np.full((Nu, Nu), np.nan, dtype=float)
RE_TB_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
LOG_ARG_TB= np.full((Nu, Nu), np.nan, dtype=float)
LOG_ARG_TB_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
LAMBDA_TB= np.full((Nu, Nu), np.nan, dtype=float)
F_T_BAR= np.full((Nu, Nu), np.nan, dtype=float)

P_TB_B_BAR= np.full((Nu, Nu), np.nan, dtype=float)
DP_AN_TB_BAR= np.full((Nu, Nu), np.nan, dtype=float)
W_G_INJ= np.full((Nu, Nu), np.nan, dtype=float)

U_AVG_L_BH= np.full((Nu, Nu), np.nan, dtype=float)
RE_BH= np.full((Nu, Nu), np.nan, dtype=float)
LOG_ARG_BH= np.full((Nu, Nu), np.nan, dtype=float)
LAMBDA_BH= np.full((Nu, Nu), np.nan, dtype=float)
F_BH_BAR= np.full((Nu, Nu), np.nan, dtype=float)
P_BH_BAR= np.full((Nu, Nu), np.nan, dtype=float)

DP_RES_BH_BAR= np.full((Nu, Nu), np.nan, dtype=float)
W_RES= np.full((Nu, Nu), np.nan, dtype=float)
W_L_RES= np.full((Nu, Nu), np.nan, dtype=float)
W_G_RES= np.full((Nu, Nu), np.nan, dtype=float)
RHO_G_TB_B= np.full((Nu, Nu), np.nan, dtype=float)

DENOM_ALPHA_B= np.full((Nu, Nu), np.nan, dtype=float)
DENOM_ALPHA_B_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
ALPHA_L_TB_B= np.full((Nu, Nu), np.nan, dtype=float)
ALPHA_L_TB_T= np.full((Nu, Nu), np.nan, dtype=float)

RHO_MIX_TB_T= np.full((Nu, Nu), np.nan, dtype=float)
RHO_MIX_TB_T_SAFE= np.full((Nu, Nu), np.nan, dtype=float)

DP_TB_CHOKE_BAR= np.full((Nu, Nu), np.nan, dtype=float)
W_OUT= np.full((Nu, Nu), np.nan, dtype=float)
Q_OUT= np.full((Nu, Nu), np.nan, dtype=float)
DENOM_ALPHA_T= np.full((Nu, Nu), np.nan, dtype=float)
DENOM_ALPHA_T_SAFE= np.full((Nu, Nu), np.nan, dtype=float)
ALPHA_G_TB_T= np.full((Nu, Nu), np.nan, dtype=float)
W_G_TB_T= np.full((Nu, Nu), np.nan, dtype=float)
W_G_OUT= np.full((Nu, Nu), np.nan, dtype=float)
W_L_OUT= np.full((Nu, Nu), np.nan, dtype=float)

RES= np.full((Nu, Nu), np.nan, dtype=float)

STABLE = np.full((Nu, Nu), np.nan, dtype=float)  # 1 stable, 0 unstable, NaN unknown/fail

y0_fixed = np.array([3919.7688, 437.16663, 7956.1206], dtype=float)

model = build_steady_state_model(glc_casadi, state_size=3, control_size=2, name="glc")

RES_TOL = 1e-6

for i, u1 in enumerate(u_grid):
    for j, u2 in enumerate(u_grid):
        print("\n----------------------------------")
        print(f"u1={u1} u2={u2}")
        try:
            y_star, dx_star, z_star, eig, stable, stats = solve_equilibrium_ipopt(
                model=model,
                u_val=[u1, u2],
                y_guess=y0_fixed,
            )
            print("IPOPT macro status:", stats["success"])
            print("IPOPT status:", stats["return_status"])
            success = bool(stats.get("success", False))
            if success:
                print("Solver succeeded!")
            else:
                print("Solver failed!")
                continue
            dx_np = np.array(dx_star, dtype=float).squeeze()
            res = float(np.linalg.norm(dx_np))
            RES[i, j] = res
            if res > RES_TOL:
                print(f"High Residual {res:.3e}>{RES_TOL:.1e})")
                continue
            if stable:
                u1_stable.append(u1)
                u2_stable.append(u2)
            else:
                u1_unstab.append(u1)
                u2_unstab.append(u2)
            y_np = np.array(y_star, dtype=float).squeeze()
            z_np = np.array(z_star, dtype=float).squeeze()

            P_AN_T_BAR[i,j] = z_np[0]
            P_AN_B_BAR[i,j] = z_np[1]
            RHO_G_AN_B[i,j] = z_np[2]
            RHO_G_IN[i,j] = z_np[3]

            DP_GS_AN_BAR[i,j] = z_np[4]
            W_G_IN_ORIGINAL[i,j] = z_np[5]
            W_G_IN[i,j] = z_np[6]

            V_GAS_TB_T[i,j] = z_np[7]
            V_GAS_TB_T_SAFE[i,j] = z_np[8]
            RHO_G_TB_T[i,j] = z_np[9]
            P_TB_T_BAR[i,j] = z_np[10]

            RHO_AVG_MIX_TB[i,j] = z_np[11]
            ALPHA_AVG_L_TB[i,j] = z_np[12]
            ALPHA_G_TB_B[i,j]= z_np[13]
            U_AVG_L_TB[i,j] = z_np[14]
            DENOM_G[i,j] = z_np[15]
            DENOM_G_SAFE[i,j]= z_np[16]
            U_AVG_G_TB[i,j] = z_np[17]
            U_AVG_MIX_TB[i,j] = z_np[18]

            RE_TB[i,j] = z_np[19]
            RE_TB_SAFE[i,j] = z_np[20]
            LOG_ARG_TB[i,j] = z_np[21]
            LOG_ARG_TB_SAFE[i,j]  = z_np[22]
            LAMBDA_TB[i,j]  = z_np[23]
            F_T_BAR[i,j] = z_np[24]

            P_TB_B_BAR[i,j]  = z_np[25]
            DP_AN_TB_BAR[i,j]  = z_np[26]
            W_G_INJ[i,j]  = z_np[27]

            U_AVG_L_BH[i,j]  = z_np[28]
            RE_BH[i,j]  = z_np[29]
            LOG_ARG_BH[i,j]  = z_np[30]
            LAMBDA_BH[i,j]  = z_np[31]
            F_BH_BAR[i,j]  = z_np[32]
            P_BH_BAR[i,j]  = z_np[33]

            DP_RES_BH_BAR[i,j]  = z_np[34]
            W_RES[i,j]  = z_np[35]
            W_L_RES[i,j]  = z_np[36]
            W_G_RES[i,j]  = z_np[37]
            RHO_G_TB_B[i,j]  = z_np[38]

            DENOM_ALPHA_B[i,j]  = z_np[39]
            DENOM_ALPHA_B_SAFE[i,j]  = z_np[40]
            ALPHA_L_TB_B[i,j]  = z_np[41]
            ALPHA_L_TB_T[i,j]  = z_np[42]

            RHO_MIX_TB_T[i,j]  = z_np[43]
            RHO_MIX_TB_T_SAFE[i,j]  = z_np[44]

            DP_TB_CHOKE_BAR[i,j]  = z_np[45]
            W_OUT[i,j]  = z_np[46]
            Q_OUT[i,j]  = z_np[47]
            DENOM_ALPHA_T[i,j]  = z_np[48]
            DENOM_ALPHA_T_SAFE[i,j]  = z_np[49]
            ALPHA_G_TB_T[i,j]  = z_np[50]
            W_G_OUT[i,j]  = z_np[51]
            W_L_OUT[i,j]  = z_np[52]

            print("y*:", y_np, "res:", res, "stable:", stable)

        except Exception as e:
            print("Exception:", repr(e))
            continue


from matplotlib.colors import Normalize

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
        U1, U2, Zm,
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
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)





fig1 = plt.figure(figsize=(16, 18))

axes = [
    fig1.add_subplot(2, 2, 1, projection="3d"),
    fig1.add_subplot(2, 2, 2, projection="3d"),
    fig1.add_subplot(2, 2, 3, projection="3d"),
    fig1.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig1,axes[0], U1, U2, P_AN_T_BAR, "p_an_t(u1,u2)", "bar")
plot_surface(fig1,axes[1], U1, U2, P_AN_B_BAR, "p_an_b(u1,u2)", "bar")

# Row 2
plot_surface(fig1,axes[2], U1, U2, P_TB_B_BAR,  "p_tb_b(u1,u2)", "bar")
plot_surface(fig1,axes[3], U1, U2, P_BH_BAR, "p_bh(u1,u2)", "bar")

plt.tight_layout()
plt.show()

# # Row 1
# plot_surface(fig,axes[0], U1, U2, WOUT, "w_out(u1,u2)", "kg/s")
# plot_surface(fig,axes[1], U1, U2, WINJ, "w_G_inj(u1,u2)", "kg/s")
#
# # Row 2
# plot_surface(fig,axes[2], U1, U2, PBH,  "P_bh_bar(u1,u2)", "bar")
# plot_surface(fig,axes[3], U1, U2, ALPH, "alpha_avg_L_tb(u1,u2)", "-")
#
# # Row 3
# plot_surface(fig,axes[4], U1, U2, FT,   "F_t_bar(u1,u2)", "bar")
# plot_surface(fig,axes[5], U1, U2, WRES,   "w_res(u1,u2)", "kg/s")
#
# # Row 4
# plot_surface(fig,axes[6], U1, U2, PTBT, "P_tb_t_bar(u1,u2)", "bar")
# plot_surface(fig,axes[7], U1, U2, DPCH, "dP_tb_choke_bar(u1,u2)", "bar")
#


plt.figure()
if len(u1_stable) > 0:
    plt.plot(u1_stable, u2_stable, "o", label="stable (O)")
if len(u1_unstab) > 0:
    plt.plot(u1_unstab, u2_unstab, "x", label="unstable (X)")
plt.xlabel("u1")
plt.ylabel("u2")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.grid(True)
# plt.legend()
plt.title("Stability map in (u1,u2) plane (solver failures not shown)")
plt.show()