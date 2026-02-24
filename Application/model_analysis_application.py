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
SIM_KIND="surrogate" # <-- change to "surrogate" or "rigorous"

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

################################################################
fig1 = plt.figure(figsize=(16, 18))
fig1.suptitle("MacroPressures surfaces vs Gas Lift Controls", fontsize=16)

axes = [
    fig1.add_subplot(2, 2, 1, projection="3d"),
    fig1.add_subplot(2, 2, 2, projection="3d"),
    fig1.add_subplot(2, 2, 3, projection="3d"),
    fig1.add_subplot(2, 2, 4, projection="3d")
]
plot_surface(fig1,axes[0], U1, U2, results["OUT"]["P_an_t_bar"], "p_an_t(u1,u2)", "bar")
plot_surface(fig1,axes[1], U1, U2, results["OUT"]["P_an_b_bar"], "p_an_b(u1,u2)", "bar")
plot_surface(fig1,axes[2], U1, U2, results["OUT"]["P_tb_b_bar"],  "p_tb_b(u1,u2)", "bar")
plot_surface(fig1,axes[3], U1, U2, results["OUT"]["P_bh_bar"], "p_bh(u1,u2)", "bar")
for ax in axes:
    ax.view_init(elev=45, azim=45)
plt.tight_layout()
plt.show()

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

# #################################################################
fig3 = plt.figure(figsize=(16, 18))
fig3.suptitle("History behind the bottom hole pressure", fontsize=16)

axes = [
    fig3.add_subplot(2, 2, 1, projection="3d"),
    fig3.add_subplot(2, 2, 2, projection="3d"),
    fig3.add_subplot(2, 2, 3, projection="3d"),
    fig3.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig3,axes[0], U1, U2, results["OUT"]["P_bh_bar"],  "p_bh(u1,u2)", "bar")
plot_surface(fig3,axes[1], U1, U2, results["OUT"]["P_tb_b_bar"], "p_tb_b(u1,u2)", "bar")
plot_surface(fig3,axes[2],U1,U2,results["OUT"]["rho_mix_bh"],"p_hidrostatic","bar")
plot_surface(fig3,axes[3], U1, U2, results["OUT"]["F_bh_bar"], "F_bh(u1,u2)", "bar")
axes[0].view_init(elev=25, azim=45)
axes[1].view_init(elev=25,azim=45)# opposite view
axes[2].view_init(elev=25,azim=45)# opposite view
axes[3].view_init(elev=45,azim=45)# opposite view
plt.tight_layout()
plt.show()

# #################################################################

fig4 = plt.figure(figsize=(16, 18))
fig4.suptitle("Zoom in the friction term", fontsize=16)

axes = [
    fig4.add_subplot(2, 2, 1, projection="3d"),
    fig4.add_subplot(2, 2, 2, projection="3d"),
    fig4.add_subplot(2, 2, 3, projection="3d"),
    fig4.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig4,axes[0], U1, U2, results["OUT"]["F_t_bar"], "F_t(u1,u2)", "bar")
plot_surface(fig4,axes[1], U1, U2, results["OUT"]["alpha_avg_L_tb"],  "alpha_avg(u1,u2)", "bar")
plot_surface(fig4,axes[2], U1, U2, results["OUT"]["Re_tb"], "Re_tb(u1,u2)", "bar")
plot_surface(fig4,axes[3], U1, U2, results["OUT"]["U_avg_G_tb"], "U_g(u1,u2)", "bar")

for ax in axes:
    ax.view_init(elev=45, azim=45)    # opposite view

plt.tight_layout()
plt.show()

# #################################################################

fig5 = plt.figure(figsize=(16, 18))
fig5.suptitle("Reservoir Flows", fontsize=16)

axes = [
    fig5.add_subplot(2, 2, 1, projection="3d"),
    fig5.add_subplot(2, 2, 2, projection="3d"),
    fig5.add_subplot(2, 2, 3, projection="3d"),
    fig5.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig5,axes[0], U1, U2, results["OUT"]["w_res"],  "w_res(u1,u2)", "kg/s")
plot_surface(fig5,axes[1], U1, U2, results["OUT"]["w_L_res"], "w_l_res(u1,u2)", "kg/s")
plot_surface(fig5,axes[2], U1, U2, results["OUT"]["w_G_res"], "w_g_res(u1,u2)", "kg/s")
plot_surface(fig5,axes[3], U1, U2, results["OUT"]["w_out"], "w_out(u1,u2)", "kg/s")
for ax in axes:
    ax.view_init(elev=45, azim=45)  # opposite view

plt.tight_layout()
plt.show()

# #################################################################

fig6 = plt.figure(figsize=(16, 18))
fig6.suptitle("Flows", fontsize=16)
axes = [
    fig6.add_subplot(2, 2, 1, projection="3d"),
    fig6.add_subplot(2, 2, 2, projection="3d"),
    fig6.add_subplot(2, 2, 3, projection="3d"),
    fig6.add_subplot(2, 2, 4, projection="3d")
]
plot_surface(fig6,axes[0], U1, U2, results["OUT"]["w_G_res"],  "w_g_res(u1,u2)", "kg/s")
plot_surface(fig6,axes[1], U1, U2, results["OUT"]["w_G_inj"], "w_g_inj(u1,u2)", "kg/s")
plot_surface(fig6,axes[2], U1, U2, results["OUT"]["w_L_res"], "w_g_out(u1,u2)", "kg/s")
plot_surface(fig6,axes[3], U1, U2, results["OUT"]["Q_out"], "Q_out(u1,u2)", "kg/s")
for ax in axes:
    ax.view_init(elev=45, azim=45)   # opposite view
plt.tight_layout()
plt.show()


# #################################################################

fig7 = plt.figure(figsize=(16, 18))
fig7.suptitle("Delta pressure in the chokes", fontsize=16)

axes = [
    fig7.add_subplot(2, 2, 1, projection="3d"),
    fig7.add_subplot(2, 2, 2, projection="3d"),
    fig7.add_subplot(2, 2, 3, projection="3d"),
    fig7.add_subplot(2, 2, 4, projection="3d")
]

plot_surface(fig7,axes[0], U1, U2, results["OUT"]["dP_gs_an_bar"],  "DP_GS_AN(u1,u2)", "bar")
plot_surface(fig7,axes[1], U1, U2,results["OUT"]["dP_an_tb_bar"], "DP_AN_TB(u1,u2)", "bar")
plot_surface(fig7,axes[2], U1, U2, results["OUT"]["dP_tb_choke_bar"], "DP_TB_CHOKE(u1,u2)", "bar")
plot_surface(fig7,axes[3], U1, U2, results["OUT"]["dP_res_bh_bar"], "DP_RES_BH(u1,u2)", "bar")
for ax in axes:
    ax.view_init(elev=45, azim=45)   # opposite view
plt.tight_layout()
plt.show()

# #################################################################

fig8 = plt.figure(figsize=(16, 18))
fig8.suptitle("Production", fontsize=16)
axes = [
    fig8.add_subplot(2, 2, 1, projection="3d"),
    fig8.add_subplot(2, 2, 2, projection="3d"),
    fig8.add_subplot(2, 2, 3, projection="3d"),
    fig8.add_subplot(2, 2, 4, projection="3d")
]
plot_surface(fig8,axes[0], U1, U2, results["OUT"]["w_out"], "Total production", "kg/s")
plot_surface(fig8,axes[1], U1, U2, results["OUT"]["w_L_out"], "Production of liquid", "kg/s")
plot_surface(fig8,axes[2], U1, U2, results["OUT"]["w_o_out"], "Production of oil", "kg/s")
plot_surface(fig8,axes[3], U1, U2, results["OUT"]["w_w_out"], "Production of water`", "kg/s")
for ax in axes:
    ax.view_init(elev=45, azim=45)   # opposite view
plt.tight_layout()
plt.show()

# #################################################################



def plot_stability_map(U1,
                       U2,
                       STABLE,
                       title="Stability map in (u1,u2) plane",
                       xlim=(0.0, 1.0), ylim=(0.2, 1.0)):
    """
        U1, U2: meshgrid arrays (same shape)
        STABLE: same shape, with values:
            1.0  -> stable
            0.0  -> unstable
            NaN  -> failure/unknown
        """
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
                   s=90,
                   facecolors="none",
                   edgecolors="green",
                   linewidths=1.5,
                   label="stable")
    if np.any(unstab_mask):
        ax.scatter(U1[unstab_mask],
                   U2[unstab_mask],
                   marker='x',
                   s=90,
                   c="red",
                   linewidths=1.5,
                   label="unstable"
                   )
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.35)
    ax.set_title(title)

    # legend only if there are labeled items
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    return ax

# def extract_stability_boundary_from_grid(u1_grid,u2_grid,STABLE):
#     u1_grid=np.asarray(u1_grid,dtype=float).reshape(-1)
#     u2_grid=np.asarray(u2_grid,dtype=float).reshape(-1)
#     STABLE=np.asarray(STABLE,dtype=float)
#
#     Nu1=len(u1_grid)
#     Nu2=len(u2_grid)

ax=plot_stability_map(U1,
                      U2,
                      STABLE=results["STABLE"],
                      title="Stability map + fitted stability constraint")
plt.tight_layout()
plt.show()