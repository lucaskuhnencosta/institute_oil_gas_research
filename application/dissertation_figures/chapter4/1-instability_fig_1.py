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

######################################################################################
######################################################################################
######################################################################################
######################################################################################
def fake_binary_stability_search(results):
    STABLE_FULL = np.asarray(results["STABLE"], dtype=float)
    U1 = np.asarray(results["U1"], dtype=float)
    U2 = np.asarray(results["U2"], dtype=float)
    Nu1, Nu2 = STABLE_FULL.shape
    STABLE_FAKE = np.full_like(STABLE_FULL, np.nan, dtype=float)
    CONSULTED = np.zeros_like(STABLE_FULL, dtype=bool)
    frontier = []
    def consult(i, j):
        CONSULTED[i, j] = True
        STABLE_FAKE[i, j] = STABLE_FULL[i, j]
        return STABLE_FULL[i, j]
    i_start=-1
    for i in range(Nu1):
        value_right = consult(Nu2-1, i)
        if value_right == 1.0:
            break

        left = i_start+1
        right = Nu2 - 2

        value_left = consult(left, i)
        if value_left == 0.0:
            continue
        j_stable = left

        while right - left > 1:
            mid = (left + right) // 2
            value_mid = consult(mid, i)

            if value_mid == 1.0:
                left = mid
                j_stable = mid
            elif value_mid == 0.0:
                right = mid
            else:
                # If the full sweep has NaN/failure, register it but do not trust it.
                # For now, move right boundary conservatively.
                right = mid
        i_start=j_stable
        # Fill inferred classification from monotonicity
        # STABLE_FAKE[i, :j_stable + 1] = 1.0
        # STABLE_FAKE[i, j_stable + 1:] = 0.0

        # frontier.append((i, j_stable, U1[i, j_stable], U2[i, j_stable]))

    return {
        # "STABLE_FAKE": STABLE_FAKE,
        "CONSULTED": CONSULTED,
        # "frontier": frontier,
    }


######################################################################################
######################################################################################
######################################################################################
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

print(results["STABLE"])
fake_results = fake_binary_stability_search(results)
print(fake_results["CONSULTED"])

ax = plot_stability_map(
    U1=results["U1"],
    U2=results["U2"],
    STABLE=results["STABLE"],
    CONSULTED=fake_results["CONSULTED"],
    U1_MIN=U1_MIN,
    U2_MIN=U2_MIN,
    title=None,
)

fig = ax.figure

fig.tight_layout()
fig.savefig("stability_plot_1.pdf", format="pdf", bbox_inches="tight")
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