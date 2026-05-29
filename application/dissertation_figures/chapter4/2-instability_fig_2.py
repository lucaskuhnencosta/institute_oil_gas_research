from configuration.wells import get_wells
from application.simulation_engine import *
from application.plotting_engine import *

from settings import *

##################################################################################3
######################################################################################
wells=get_wells()
well="P1"
params=wells[well]
MODE = "rigorous"
U1_MIN=0.05 #if you ever change this, you need to change inside the black box optimizer
U2_MIN=0.10 #if you ever change this, you need to change inside the black box optimizer
##################################################################################3
######################################################################################



coeff_stability={}

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

results = run_sweep(model_rig,
                              U1_MIN=U1_MIN,
                              U2_MIN=U2_MIN,
                              U_SIM_SIZE=U_SIM_SIZE,
                              y_guess_init=y_guess_rig,
                              z_guess_init=z_guess_rig,
                              RES_TOL_DX=RES_TOL_DX,
                              TOL_EIG=TOL_EIG)

boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(
    results["u1_grid"],
    results["u2_grid"],
    results["STABLE"]
)
b_hat = fit_boundary_polynomial(boundary_u1,
                                boundary_u2,
                                deg=degree_polynomial)

ax = plot_stability_map(
    U1=results["U1"],
    U2=results["U2"],
    STABLE=results["STABLE"],
    CONSULTED=None,
    U1_MIN=U1_MIN,
    U2_MIN=U2_MIN,
    title=None,
    show_boundary=True,
    u1_grid=results["u1_grid"],
    u2_grid=results["u2_grid"],
)
overlay_boundary_and_fit(ax,
                         b_hat)
fig = ax.figure

fig.tight_layout()
fig.savefig("stability_plot_2.pdf", format="pdf", bbox_inches="tight")
plt.show()
#
# ax = plot_stability_map(U1=results["U1"],
#                         U2=results["U2"],
#                         STABLE=results["STABLE"],
#                         U1_MIN=U1_MIN,
#                         U2_MIN=U2_MIN,
#                         title=f"Mapa de estabilidade para o poço {well}")
#

# overlay_boundary_and_fit(ax,
# #                          b_hat)
# plt.tight_layout()

# save_dir = os.path.join("results", "figures", "stability_maps")
# os.makedirs(save_dir, exist_ok=True)
#
# filename = f"stability_map_{well}.pdf"
# filepath = os.path.join(save_dir, filename)

# ax.figure.savefig(filepath, format="pdf", bbox_inches="tight")
# plt.show()