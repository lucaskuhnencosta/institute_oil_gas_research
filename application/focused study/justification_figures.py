import numpy as np
import matplotlib.pyplot as plt
tiny=1e-12
eps=1e-12
k_pos=50
dp_pa=1e5


def softplus_stable(x):
    # softplus(x) = max(x,0) + log(1+exp(-|x|)), stable for large x
    ax = np.sqrt(x * x + tiny)
    return 0.5 * (x + ax) + np.log(1 + np.exp(-ax))


def smooth_pos_scaled(dp_pa, scale=1):
    x = (k_pos * dp_pa) / scale
    # softplus(z) = (1/k) log(1+exp(k z))
    return (scale / k_pos) * (softplus_stable(x))

K_inj=1.40e-4
rho_G_in=80
# dp range: 0 to 100000 Pa
dp = np.linspace(-0.5, 0.5, 1000)

smooth=smooth_pos_scaled(dp,scale=1)
nonsmooth= np.maximum(dp, 0)

# Plot both on the same figure
plt.figure()
plt.plot(dp, smooth, label="smooth_pos_scaled")
plt.plot(dp, nonsmooth, label="max(dp, 0)")
plt.xlabel("dp (Pa)")
plt.ylabel("w_G_in")
plt.title("Comparison: Smooth vs max(dp, 0)")
plt.legend()
plt.show()