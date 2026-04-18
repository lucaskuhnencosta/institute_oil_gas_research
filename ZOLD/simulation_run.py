from networks.networks import PINN, AlgNN
from utilities.block_builders import *
from optimization.production_optimizer_NN import optimize_single_well_production_NN
from optimization.production_optimizer import optimize_field_production
import numpy as np
from solvers.steady_state_solver import solve_equilibrium_ipopt
from application.simulation_engine import make_model
from optimization.surrogate_based_optimization import SBO

####################################################################################
# IMPORTING AND BUILDING THE PINN
####################################################################################

pinn = PINN(
    hidden_units=[64,64,64]
)
algnn = AlgNN(
    hidden_units=[64,64,64,64]
)
pinn = load_model_weights(pinn, "../training/PINN.pth")
algnn  = load_model_weights(algnn, "../training/AlgNN.pth")
pinn_w = extract_pinn_standard_weights(pinn)
alg_w  = extract_algnn_standard_weights(algnn)
pinn_y_min = pinn.y_min.cpu().numpy().tolist()
pinn_y_max = pinn.y_max.cpu().numpy().tolist()
pinn_u_min = pinn.u_min.cpu().numpy().tolist()
pinn_u_max = pinn.u_max.cpu().numpy().tolist()
alg_y_min = algnn.y_min.cpu().numpy().tolist()
alg_y_max = algnn.y_max.cpu().numpy().tolist()
alg_u_min = algnn.u_min.cpu().numpy().tolist()
alg_u_max = algnn.u_max.cpu().numpy().tolist()
alg_z_min = algnn.z_min.cpu().numpy().tolist()
alg_z_max = algnn.z_max.cpu().numpy().tolist()
F_pinn = build_casadi_pinn_function_standard(pinn_w,
                                                 pinn_y_min,
                                                 pinn_y_max,
                                                 pinn_u_min,
                                                 pinn_u_max)
F_u2z = build_casadi_surrogate_u2z(
    pinn_weights=pinn_w,
    algnn_weights=alg_w,
    pinn_y_min=pinn_y_min, pinn_y_max=pinn_y_max,
    pinn_u_min=pinn_u_min, pinn_u_max=pinn_u_max,
    alg_y_min=alg_y_min,   alg_y_max=alg_y_max,
    alg_u_min=alg_u_min,   alg_u_max=alg_u_max,
    alg_z_min=alg_z_min,   alg_z_max=alg_z_max,
)

################################################
# Starting the simulation
################################################


P_list = np.array([88,90,92,94,96,98,100,102,104,104,106,106,106,108,110,112,114,116,118,120,122,124,126,128], dtype=float)
hours = np.linspace(0, 24, len(P_list))

u_PINN = np.zeros((len(P_list), 2))
u_SBO  = np.zeros((len(P_list), 2))

m_o_out_PINN = np.zeros(len(P_list))
P_bh_PINN    = np.zeros(len(P_list))

m_o_out_SBO = np.zeros(len(P_list))
P_bh_SBO    = np.zeros(len(P_list))
for i in range(len(P_list)):

    P_min_bh_bar=P_list[i]
    # 1) PINN CONTROLS
    res = optimize_single_well_production_NN(
        F_u2z=F_u2z,
        F_pinn=F_pinn,
        u_guess=(0.8, 0.8),
        P_min_bh_bar=P_min_bh_bar,
        P_max_tb_b_bar=120,
    )

    print("Solver success:", res["stats"]["success"])
    print("u* =", res["u_star"])
    print("y* =", res["y_star"])
    print("m_o_out* =", res["m_o_out"])
    print("P_bh* =", res["P_bh"])
    print("P_tb_b* =", res["P_tb_b"])

    u_PINN[i,:]=res["u_star"]
    u1=res["u_star"][0]
    u2=res["u_star"][1]

    # 2) EFFECT OF THESE CONTROLS IN REAL LIFE

    y_guess_rig = [3679.08033973,
                   289.73390193,
                   3167.56224658,
                   1041.96126532,
                   50.46858403,
                   759.52720527,
                   249.84447542]

    z_guess_rig = [8.75897957e+06,
    8.42155186e+06,
    2.17230613e+01,
    2.17230613e+01]

    print('\n\n')
    print(f"Now we take this control and apply to the plant...")
    model=make_model("rigorous",BSW=0.20,GOR=0.05,PI=3.0e-6)
    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
        model=model,
        u_val=[u1, u2],
        y_guess=y_guess_rig,
        z_guess=z_guess_rig
    )
    m_o_out_PINN[i]=out_star[38]
    P_bh_PINN[i]=out_star[15]

    #########################
    # 1) SURROGATE BASED OPTIMIZATION CONTROLS

    u_converged=SBO(P_min_bh_bar)
    u_SBO[i,:]=u_converged["u_converged"]
    u1=u_converged["u_converged"][0]
    u2=u_converged["u_converged"][1]

    # 2) EFFECT OF THESE CONTROLS IN REAL LIFE

    y_guess_rig = [3679.08033973,
                   289.73390193,
                   3167.56224658,
                   1041.96126532,
                   50.46858403,
                   759.52720527,
                   249.84447542]

    z_guess_rig = [8.75897957e+06,
    8.42155186e+06,
    2.17230613e+01,
    2.17230613e+01]

    print('\n\n')
    print(f"Now we take this control and apply to the plant...")
    model=make_model("rigorous",BSW=0.20,GOR=0.05,PI=3.0e-6)
    y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
        model=model,
        u_val=[u1, u2],
        y_guess=y_guess_rig,
        z_guess=z_guess_rig
    )
    P_bh_SBO[i]=out_star[15]
    m_o_out_SBO[i]=out_star[38]


print(u_PINN)
print(m_o_out_PINN)
print(P_bh_PINN)

print(u_SBO)
print(m_o_out_SBO)
print(P_bh_SBO)
P_list = np.array(P_list, dtype=float)

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Convert P_list to numpy, just in case
# ------------------------------------------------------------
P_list = np.array(P_list, dtype=float)
total_oil_PINN = np.sum(m_o_out_PINN)*60*60
total_oil_SBO  = np.sum(m_o_out_SBO)*60*60
price_per_kg = 0.70

revenue_PINN = total_oil_PINN * price_per_kg
revenue_SBO  = total_oil_SBO  * price_per_kg

extra_profit = revenue_SBO - revenue_PINN
# ------------------------------------------------------------
# Colors
# matplotlib uses 0-1 RGB(A), so divide by 255
# ------------------------------------------------------------
color_u1   = (252/255, 77/255, 45/255)    # rgba(252, 77, 45)
color_u2   = (21/255, 59/255, 131/255)    # rgba(21, 59, 131)
color_moil = "black"
color_pbh  = (0/255, 132/255, 77/255)     # rgba(0, 132, 77)

# ------------------------------------------------------------
# Figure + axes
# 3 rows x 2 cols:
#   left  = PINN
#   right = SBO
# ------------------------------------------------------------
fig, axs = plt.subplots(
    3, 2,
    figsize=(14, 10),
    sharex='col',
    gridspec_kw={"height_ratios": [1.1, 1.1, 1.2]}
)

# ------------------------------------------------------------
# Old MATLAB-like style: gray background
# ------------------------------------------------------------
fig.patch.set_facecolor("white")

for ax_row in axs:
    for ax in ax_row:
        ax.set_facecolor("white")
        ax.grid(True, color="white", linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.tick_params(colors="black")

fig.suptitle(
    "Simulation of 24 hours of Real-Time optimization of Gas Lift Well",
    fontsize=18,
    fontweight="bold"
)

fig.text(
    0.28,0.70,
    f"Revenue ≈ ${revenue_SBO:,.0f} / day",
    fontsize=16,
    fontweight="bold",
    ha="center"
)

fig.text(
    0.68,0.70,
    f"Revenue ≈ ${revenue_PINN:,.0f} / day",
    fontsize=16,
    fontweight="bold",
    ha="center"
)


# ============================================================
# LEFT COLUMN: PINN
# ============================================================

# Top: oil
axs[0, 0].plot(
    hours, m_o_out_PINN,
    color=color_moil,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label=r"$m_{oil,out}$"
)
axs[0, 0].set_title("Oil Production with Standalone PINN Model", fontsize=13, fontweight="bold")
axs[0, 0].set_ylabel(r"$m_{oil,out}$", fontsize=11)
axs[0, 0].legend(loc="best", frameon=True)

# Middle: P_bh
axs[1, 0].plot(
    hours, P_bh_PINN,
    color=color_pbh,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label=r"$P_{bh}$ [bar] (plant)"
)
axs[1, 0].plot(
    hours, P_list,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=r"$P_{bh}^{min}$ [bar] target"
)
axs[1, 0].set_ylabel(r"$P_{bh}$ [bar]", fontsize=11)
axs[1, 0].legend(loc="best", frameon=True)
axs[1,0].set_title("Bottom-Hole Pressure (Plant)")

# Bottom: controls
axs[2, 0].plot(
    hours, u_PINN[:, 0],
    color=color_u1,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Production Choke"
)
axs[2, 0].plot(
    hours, u_PINN[:, 1],
    color=color_u2,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Gas lift choke"
)
axs[2, 0].set_ylabel("Controls", fontsize=11)
axs[2, 0].set_xlabel("Time [hours]", fontsize=11)
axs[2, 0].legend(loc="best", frameon=True)
axs[2,0].set_title("Choke Controls")
# ============================================================
# RIGHT COLUMN: SBO
# ============================================================

# Top: oil
axs[0, 1].plot(
    hours, m_o_out_SBO,
    color=color_moil,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label=r"$m_{oil,out}$"
)
axs[0, 1].set_title("Oil Production with PINN Model + Surrogate-based optimization", fontsize=13, fontweight="bold")
axs[0, 1].set_ylabel(r"$m_{oil,out}$", fontsize=11)
axs[0, 1].legend(loc="best", frameon=True)

# Middle: P_bh
axs[1, 1].plot(
    hours, P_bh_SBO,
    color=color_pbh,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label=r"$P_{bh}$ [bar] (plant)"
)
axs[1, 1].plot(
    hours, P_list,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=r"$P_{bh}^{min}$ [bar] target"
)
axs[1, 1].set_ylabel(r"$P_{bh}$ [bar]", fontsize=11)
axs[1, 1].legend(loc="best", frameon=True)
axs[1,1].set_title("Bottom-Hole Pressure (Plant)")


# Bottom: controls
axs[2, 1].plot(
    hours, u_SBO[:, 0],
    color=color_u1,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Production Choke"
)
axs[2, 1].plot(
    hours, u_SBO[:, 1],
    color=color_u2,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="Gas lift choke"
)
axs[2, 1].set_ylabel("Controls", fontsize=11)
axs[2, 1].set_xlabel("Time [hours]", fontsize=11)
axs[2, 1].legend(loc="best", frameon=True)
axs[2,1].set_title("Choke Controls")

axs[1, 0].annotate(
    "Constraint violation = operational risk",

    xy=(9, P_bh_PINN[9]),  # point where the arrow points (violation point)
    xytext=(2, P_bh_PINN[9] + 15),  # text location (north-west direction)

    arrowprops=dict(
        arrowstyle="->",
        linewidth=2,
        color="black"
    ),

    fontsize=11,
    fontweight="bold",
    ha="left"
)

for ax in axs.flatten():
    ax.set_xlim(0, 24)

for ax in axs.flatten():
    ax.set_xticks(np.arange(0, 25, 4))   # tick every 4 hours
# ------------------------------------------------------------
# Optional: make x ticks exactly your target pressures
# ------------------------------------------------------------
# for j in range(2):
#     axs[2, j].set_xticks(P_list)

# ------------------------------------------------------------
# Nice layout
# ------------------------------------------------------------
plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.20)

plt.savefig("real_time_optimization.png", dpi=300, bbox_inches="tight")
plt.savefig("real_time_optimization.pdf", bbox_inches="tight")

plt.show()