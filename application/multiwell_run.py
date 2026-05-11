"""
THIS IS THE ULTIMATE RUNNER FILE FOR THE ARTICLE AND POSSIBLE FOR THE DISSERTATION
"""
# Normalize RGB (divide by 255)
color_primary   = (54/255,  32/255, 229/255)   # blue
color_secondary = (240/255, 101/255, 74/255)   # orange
color_third     = (183/255, 53/255, 192/255)   # purple (saved)


import numpy as np
import matplotlib.pyplot as plt

from optimization.surrogate_based_optimization import SurrogateBasedOptimization
from configuration.surrogate_based_optimizer_configs import get_solver_configs
from configuration.wells import get_wells
from application.simulation_engine import make_model
from utilities.block_builders import build_casadi_surrogate_u2z_for_well
import json
wells=get_wells()
config=get_solver_configs()
model_type = "rigorous"

well_names = list(wells.keys())
N = len(well_names)

rigorous_models=[]
F_u2z_models=[]

for well_name in well_names:
    well_data=wells[well_name]
    rigorous_models.append(make_model(
        model_type,
        BSW=well_data["BSW"],
        GOR=well_data["GOR"],
        PI=well_data["PI"],
        K_gs=well_data["K_gs"],
        K_inj=well_data["K_inj"],
        K_pr=well_data["K_pr"]
    ))
    F_u2z_models.append(build_casadi_surrogate_u2z_for_well(well_name))

opt1=SurrogateBasedOptimization(config=config,
                               wells=wells,
                               rigorous_models=rigorous_models,
                               F_u2z_models=F_u2z_models,
                               refinement=False)

result1=opt1.solve()

print(result1["history"])

u_phase1 = result1["u_opt"]
total_wo_phase1 = result1["phi"]

opt2 = SurrogateBasedOptimization(
    config=config,
    wells=wells,
    rigorous_models=rigorous_models,
    F_u2z_models=F_u2z_models,
    refinement=True,
    u_guess=u_phase1,
    total_wo=total_wo_phase1,
)



result2 = opt2.solve()
print(result2["history"])


history = result2["history"]

print(history)

# extract sequence of accepted u_k
u_sequence = [h["u_trial"].tolist() for h in history]

with open("u_sequence_4.json", "w") as f:
    json.dump(u_sequence, f, indent=2)

step_types = [h["type"] for h in history]
print(step_types)

theta = np.array([h["theta"] for h in history], dtype=float)
phi   = np.array([h["phi"] for h in history], dtype=float)

theta_model_per_well = []
theta_constraints_per_well = []
theta_platform = []


for h in history:
    # New format
    if "theta_details" in h:
        details = h["theta_details"]

        theta_model_per_well.append(
            details.get("model_mismatch_per_well", [])
        )

        theta_constraints_per_well.append(
            details.get("well_constraint_per_well", [])
        )

        platform_dict = details.get("platform_constraints", {})
        theta_platform.append([
            platform_dict.get("G_available", 0.0),
            platform_dict.get("G_max_export", 0.0),
            platform_dict.get("W_max", 0.0),
            platform_dict.get("L_max", 0.0),
        ])

theta_model_per_well = np.array(theta_model_per_well, dtype=float)
theta_constraints_per_well = np.array(theta_constraints_per_well, dtype=float)
theta_platform = np.array(theta_platform, dtype=float)

theta_model_p1 = theta_model_per_well[:, 0]
theta_model_p2 = theta_model_per_well[:, 1]

theta_constraint_p1 = theta_constraints_per_well[:, 0]
theta_constraint_p2 = theta_constraints_per_well[:, 1]

theta_platform_gas_inj = theta_platform[:, 0]
theta_platform_gas_export = theta_platform[:, 1]
theta_platform_water = theta_platform[:, 2]
theta_platform_liquid = theta_platform[:, 3]

iters = np.arange(len(history))


# ============================
# Global style
# ============================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ============================
# Single figure (two-column friendly)
# ============================
fig, ax = plt.subplots(figsize=(5.6, 3.8))

ax.plot(theta,
        phi,
        marker='o',
        linewidth=1.8,
        markersize=5,
        color=color_primary)

for i, (x, y) in enumerate(zip(theta, phi)):
    ax.annotate(
        str(i),
        (x, y),
        xytext=(-6, 6),
        textcoords="offset points",
        fontsize=14
    )


# Down arrow (objective improves ↓)
ax.annotate(
    "melhor",
    xy=(0.04, 0.50),        # arrow tip
    xytext=(0.04, 0.70),    # text position
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->",
                    linewidth=1.5,
                    color=color_secondary),
    ha="center",
    fontsize=14,
    color=color_secondary,
    rotation=90,
    fontweight="bold"
)

# Left arrow (theta improves ←)
ax.annotate(
    "melhor",
    xy=(0.60, 0.04),
    xytext=(0.80, 0.04),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->",
                    linewidth=1.5,
                    color=color_secondary),
    va="center",
    fontsize=14,
    color=color_secondary,
    fontweight="bold"
)


ax.set_xlabel(r"Infactibilidade $\theta$",fontsize=16)
ax.set_ylabel(r"Função objetivo $\phi$",fontsize=16)
ax.set_xticklabels([])
# ax.set_yticks(np.arange(-29, -9, 3))
# ax.set_ylim(-29, -9)
ax.set_xscale("log")
ax.set_title(r"Evolução das iterações no plano $(\theta,\phi)$",fontsize=18)

ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

plt.savefig("figure_theta_phi_first.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()