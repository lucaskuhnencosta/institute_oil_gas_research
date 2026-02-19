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

K_gs=1e-3
u2=1
rho_G_in=100
# dp range: 0 to 100000 Pa
dp = np.linspace(-5, 5, 1000)

w_smooth= K_gs * u2 * np.sqrt(rho_G_in*smooth_pos_scaled(dp,scale=1))
w_max = K_gs * u2 * np.sqrt(rho_G_in * np.maximum(dp, 0))

# Plot both on the same figure
plt.figure()
plt.plot(dp, w_smooth, label="smooth_pos_scaled")
plt.plot(dp, w_max, label="max(dp, 0)")
plt.xlabel("dp (Pa)")
plt.ylabel("w_G_in")
plt.title("Comparison: Smooth vs max(dp, 0)")
plt.legend()
plt.show()