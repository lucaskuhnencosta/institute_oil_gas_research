import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Fluid properties
# ------------------------------------------------------------
rho_o = 760.0        # kg/m3
rho_w = 1000.0       # kg/m3

mu_o = 3.64e-3       # Pa.s
mu_w = 1.00e-3       # Pa.s

# ------------------------------------------------------------
# BSW grid
# ------------------------------------------------------------
BSW = np.linspace(0.0, 1.0, 200)

# ------------------------------------------------------------
# Approximations
# ------------------------------------------------------------
rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)

mu = np.exp((1.0 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))

# ------------------------------------------------------------
# Combined figure with two y-axes
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(6.5, 4))

ax1.plot(BSW, rho_L, label=r"$\rho_L$")
ax1.set_xlabel("BSW (-)")
ax1.set_ylabel(r"Liquid density $\rho_L$ (kg/m$^3$)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(BSW, mu * 1e3, linestyle="--", label=r"$\mu$")
ax2.set_ylabel(r"Liquid viscosity $\mu$ (cP)")

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

plt.title("Effective liquid properties as a function of BSW")
plt.tight_layout()

plt.savefig("fluid_properties_vs_BSW.png", dpi=300)
plt.show()