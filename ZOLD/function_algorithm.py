from optimization.surrogate_based_optimization import SBO
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


res_sbo = SBO(P_bh_min=90)
u_converged = res_sbo["u_converged"]
history = res_sbo["history"]



history["u_k"] = np.array(history["u_k"], dtype=float) if history["u_k"] else np.empty((0, 2))
history["u_trial"] = np.array(history["u_trial"], dtype=float) if history["u_trial"] else np.empty((0, 2))
history["Delta"] = np.array(history["Delta"], dtype=float)
history["theta_trial"] = np.array(history["theta_trial"], dtype=float)
history["phi_trial"] = np.array(history["phi_trial"], dtype=float)

U = np.array(history["u_k"], dtype=float)   # shape (N,2)
iters = np.arange(U.shape[0])

u1 = U[:, 0]
u2 = U[:, 1]

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Extract history
# ------------------------------------------------------------
U = np.array(history["u_k"])
theta = np.array(history["theta_trial"])
phi = np.array(history["phi_trial"])

iters = np.arange(len(theta))

u1 = U[:,0]
u2 = U[:,1]

optimal_iter = len(iters) - 1   # last iterate

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axs = plt.subplots(
    2,1,
    figsize=(8,10)
)

fig.patch.set_facecolor("white")

for ax in axs:
    ax.set_facecolor("white")
    ax.grid(True,color="white",linewidth=1.2)

# ============================================================
# TOP: control space
# ============================================================

ax = axs[0]

# trajectory
ax.plot(u1,u2,color="black",linewidth=1.5,zorder=1)

for k in iters:

    if k == optimal_iter:

        ax.scatter(
            u1[k],u2[k],
            s=420,
            facecolor=(0/255,132/255,77/255),
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label="Optimal of the rigorous model"
        )

        ax.text(
            u1[k],u2[k],
            f"{k}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=10
        )

    else:

        ax.scatter(
            u1[k],u2[k],
            s=350,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=3
        )

        ax.text(
            u1[k],u2[k],
            f"{k}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

ax.set_title("Surrogate-Based optimization Iterates in Control Space",fontsize=14,fontweight="bold")
ax.set_xlabel(r"$u_1$")
ax.set_ylabel(r"$u_2$")
ax.legend(loc="lower right")

# ============================================================
# BOTTOM: filter space
# ============================================================

ax = axs[1]

# trajectory
ax.plot(theta,phi,color="black",linewidth=1.5,zorder=1)

for k in iters:

    if k == optimal_iter:

        ax.scatter(
            theta[k],phi[k],
            s=420,
            facecolor=(0/255,132/255,77/255),
            edgecolor="black",
            linewidth=1.5,
            zorder=5
        )

        ax.text(
            theta[k],phi[k],
            f"{k}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=10
        )

    else:

        ax.scatter(
            theta[k],phi[k],
            s=350,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=3
        )

        ax.text(
            theta[k],phi[k],
            f"{k}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

ax.set_title("Surrogate-Based optimization Iterates with Filter Progression",fontsize=14,fontweight="bold")
ax.set_xlabel(r"$\theta$ (infeasibility)")
ax.set_ylabel(r"$\phi$ (objective)")

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
plt.tight_layout()

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
plt.savefig("sbo_iterates.png",dpi=300,bbox_inches="tight")
plt.savefig("sbo_iterates.pdf",bbox_inches="tight")

plt.show()