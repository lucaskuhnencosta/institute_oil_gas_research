from settings import *
import numpy as np
from application.simulation_engine import make_model
from solvers.steady_state_solver import solve_equilibrium_ipopt
from matplotlib import pyplot as plt

from configuration.wells import get_wells

wells=get_wells()

u1=0.85
u2_grid=np.linspace(0.10,1.001,40)
well="P1"
params=wells[well]

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
y_guess_init=y_guess_rig
z_guess_init=z_guess_rig


is_dae=bool(model_rig["is_dae"])

Z_NAMES=model_rig["Z_NAMES"]
n_out=len(Z_NAMES)

Nu2=len(u2_grid)

OUT = {name: np.full((Nu2), np.nan, dtype=float) for name in Z_NAMES}
RES_DX= np.full((Nu2), np.nan, dtype=float)
RES_G=np.full((Nu2), np.nan, dtype=float) if is_dae else None
STABLE= np.full((Nu2), np.nan, dtype=float)

SUCCESS = np.zeros((Nu2), dtype=bool)

i_iter = range(Nu2 - 1, -1, -1)

prev_row_rightmost_y = None
prev_row_rightmost_z = None  # only used if is_dae
# RES_TOL_DX=1e-3
# RES_TOL_G=1e-3
for i in i_iter:
    u2 = u2_grid[i]
    print("\n----------------------------------")
    print(f"u2={u2}")
    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
        model=model_rig,
        u_val=[u1, u2],
        y_guess=y_guess_rig,
        z_guess=z_guess_rig
    )
    print("IPOPT success:", bool(stats.get("success", False)))
    print("IPOPT status :", stats.get("return_status", ""))

    dx_np = np.array(dx_star, dtype=float).reshape(-1)
    res_dx=float(np.linalg.norm(dx_np))
    RES_DX[i]=res_dx

    g_np=np.array(g_star, dtype=float).reshape(-1)
    res_g=float(np.linalg.norm(g_np))
    RES_G[i]=res_g

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
            OUT[name][i] = out_np[k]

    STABLE[i] = 1.0 if stable else 0.0
    SUCCESS[i] = True

    if stable:
        y_guess=np.array(y_star,dtype=float).reshape(-1)
        z_guess=np.array(z_star,dtype=float).reshape(-1)

    print("Accepted. y*:",y_star,"||dx||:", res_dx, "stable:", stable)


print(OUT['w_o_out'])

fig,ax = plt.subplots(figsize=(7.0, 4.2))

ax.plot(u2_grid,
        OUT['w_o_out'],
        color='black')

# Axis labels
ax.set_xlabel('$u_2$')
ax.set_ylabel('Surface oil rate $w_{o,out}$ (kg/s)')

ax.set_ylim(12.60, 14.60)
ax.set_xlim(0.02, 1.02)

# Optimum location
u2_opt = u2_grid[np.argmax(OUT["w_o_out"])]
wo_opt = np.max(OUT["w_o_out"])

# Highlight interval around optimum
width = 0.20
height=0.15
x0 = u2_opt - width
y0 = wo_opt - height/2
# height = ax.get_ylim()[1] - ax.get_ylim()[0]

import matplotlib.patches as patches

rect = patches.Rectangle(
    (x0, y0),
    width,
    height,
    facecolor="yellow",
    alpha=0.25,
    edgecolor=None,
    zorder=0,
)

ax.add_patch(rect)

# Make sure the curve stays above the rectangle
ax.plot(
    u2_grid,
    OUT["w_o_out"],
    color="black",
    zorder=2,
)

# Annotation below the highlighted region
ax.annotate(
    "optimal region",
    xy=(x0+(width/2), y0),
    xytext=(x0+(width/2), y0-0.30),
    ha="center",
    va="bottom",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0,
        color="black",
    ),
)


plt.tight_layout()
plt.show()
fig.savefig("w_o_out_u_2.pdf")

    # return {
    #     "OUT": OUT,
    #     "RES_DX": RES_DX,
    #     "RES_G":RES_G,
    #     "STABLE": STABLE,
    #     "SUCCESS": SUCCESS,
    #     "u1_grid": u1_grid,
    #     "u2_grid": u2_grid,
    #     "Z_NAMES": Z_NAMES,
    #     "is_dae": is_dae,
    #     "P_max": P_max
    # }
#
#
# results_all["rigorous"]