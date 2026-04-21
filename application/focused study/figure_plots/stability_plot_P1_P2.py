import matplotlib.pyplot as plt
import numpy as np

from configuration.wells import get_wells
from application.simulation_engine import make_model,run_sweep, fit_boundary_polynomial, extract_stability_boundary_from_grid
from application.plotting_engine import plot_stability_map,overlay_boundary_and_fit
############### THIS HERE IS NEVE GOING TO CHANGE ####################################
MODE = "rigorous"
RES_TOL_DX = 1e-6 # This could be a global variable
RES_TOL = 1e-6
RES_TOL_G = 1e-6 # This could be a global variable
TOL_EIG = 1e-8
small_delta=1e-2
U1_MIN=0.05 #if you ever change this, you need to change inside the black box optimizer
U2_MIN=0.10 #if you ever change this, you need to change inside the black box optimizer
U_SIM_SIZE=20
degree_polynomial = 2

results_all_wells = {}
coeff_stability = {}
######################################################################################
color_primary   = (54/255,  32/255, 229/255)   # blue
color_secondary = (240/255, 101/255, 74/255)   # orange
color_third     = (183/255, 53/255, 192/255)   # purple (saved)
######################################################################################

wells=get_wells()

fig, axes = plt.subplots(2, 1, figsize=(6.5, 11),sharex=True)

for i, (ax, well_name) in enumerate(zip(axes, ["P1", "P2"])):
    well_list = wells[well_name]
    BSW = well_list["BSW"]
    GOR = well_list["GOR"]
    PI = well_list["PI"]
    K_gs = well_list["K_gs"]
    K_inj = well_list["K_inj"]
    K_pr = well_list["K_pr"]
    y_guess_rig = well_list["y_guess_rig"]
    z_guess_rig = well_list["z_guess_rig"]

    model_rig = make_model("rigorous",
                       BSW=BSW,
                       GOR=GOR,
                       PI=PI,
                       K_gs=K_gs,
                       K_inj=K_inj,
                       K_pr=K_pr)

    y_guess = np.array(y_guess_rig, dtype=float).reshape(-1)

    results_all_wells[well_name] = run_sweep(model_rig,
                                  U1_MIN=U1_MIN,
                                  U2_MIN=U2_MIN,
                                  U_SIM_SIZE=U_SIM_SIZE,
                                  y_guess_init=y_guess_rig,
                                  z_guess_init=z_guess_rig,
                                  RES_TOL_DX=RES_TOL_DX,
                                  TOL_EIG=TOL_EIG)

    boundary_u1, boundary_u2 = extract_stability_boundary_from_grid(
        results_all_wells[well_name]["u1_grid"],
        results_all_wells[well_name]["u2_grid"],
        results_all_wells[well_name]["STABLE"]
    )
    b_hat = fit_boundary_polynomial(boundary_u1,
                                    boundary_u2,
                                    deg=degree_polynomial)
    coeff_stability[well_name] = {
        "coef": b_hat.c,
        "poly": b_hat
    }

    ax = plot_stability_map(U1=results_all_wells[well_name]["U1"],
                            U2=results_all_wells[well_name]["U2"],
                            STABLE=results_all_wells[well_name]["STABLE"],
                            U1_MIN=U1_MIN,
                            U2_MIN=U2_MIN,
                            title=f"Restrição de estabilidade para o poço {well_name}",
                            ax=ax)
    overlay_boundary_and_fit(ax,
                             b_hat)

    ax.set_aspect('equal', adjustable='box')  # <-- add this
axes[0].set_xlabel("")
axes[1].set_xlabel(r"$u_1$")


plt.tight_layout()
plt.savefig("mapa_de_estabilidade.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()
