# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import plotly.graph_objects as go
#
#
# def ksi(P_down, P_up, gamma=1.3, eps=1e-8):
#     P_up_safe = np.maximum(P_up, eps)
#     r = P_down / P_up_safe
#     val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
#
#     return val
#
# K_gs = 9.98e-5  # is the gas lift choke constant
# K_gs_new=K_gs
# T_an = 348  # K is the annulus temperature
#
# u2=np.linspace(0.1,1.001,60)
# P_at=np.linspace(90e5,145e5,60)
#
# U2, P_AT = np.meshgrid(u2, P_at)
#
# R = 8.314  # J/(K*mol) is the universal gas constant
# M_G = 0.0167  # (kg/mol) is the gas molecular weight
#
# P_gs = 140e5  # 140bar is the gas source pressure
# rho_G_in = P_gs * M_G / (R * T_an)  # 2.4
#
# scale = P_gs * np.sqrt(M_G / (R * T_an))
#
# w_G_in_atual=K_gs*U2*np.sqrt(np.maximum(rho_G_in*(P_gs-P_AT),0.0))
# w_G_in_new=K_gs_new*U2*scale*np.sqrt(np.maximum(ksi(P_AT,P_gs),0.0))
#
# # Combined 3D plot
# # =========================
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
#
# # Old model (blue)
# surf1 = ax.plot_surface(
#     U2, P_AT / 1e5, w_G_in_atual,
#     color='Blue',
#     edgecolor='none'
# )
#
# # New model (red)
# surf2 = ax.plot_surface(
#     U2, P_AT / 1e5, w_G_in_new,
#     color='Red',
#     edgecolor='none'
# )
#
# ax.set_xlabel('u2')
# ax.set_ylabel('P_at [bar]')
# ax.set_zlabel('w_G_in')
# ax.set_title('Comparison: Old vs New Valve Law')
#
# plt.tight_layout()
# plt.show()

import numpy as np
import plotly.graph_objects as go

k_pos=2000
eps = 1e-12

def softplus_stable(x):
    ax = np.sqrt(x * x + eps)
    return 0.5 * (x + ax) + np.log(1 + np.exp(-ax))


def smooth_pos_scaled(dp_pa):
    x = (k_pos * dp_pa)
    return (1 / k_pos) * (softplus_stable(x))


def smooth_max_scaled(z, zmin):
    # smooth approximation of max(z,zmin))
    return zmin + smooth_pos_scaled(z - zmin)

def ksi(P_down, P_up, gamma=1.3, eps=1e-8):
    P_up_safe = np.maximum(P_up, eps)
    r = P_down / P_up_safe
    val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
    return smooth_pos_scaled(val)

def ksi_atual(P_down, P_up, gamma=1.3, eps=1e-8):
    P_up_safe = np.maximum(P_up, eps)
    r = P_down / P_up_safe
    val = r**(2.0/gamma) - r**((gamma+1.0)/gamma)
    return val

K_gs = 9.98e-5
K_gs_new = K_gs
T_an = 348

u2 = np.linspace(0.1, 1.001, 60)
P_at = np.linspace(90e5, 145e5, 60)

U2, P_AT = np.meshgrid(u2, P_at)

R = 8.314
M_G = 0.0167
P_gs = 140e5

rho_G_in = P_gs * M_G / (R * T_an)
scale = P_gs * np.sqrt(M_G / (R * T_an))

# w_G_in_atual = K_gs * U2 * np.sqrt(rho_G_in*smooth_pos_scaled(P_gs - P_AT))
w_G_in_atual = K_gs_new * U2 * scale * np.sqrt(np.maximum(ksi_atual(P_AT, P_gs),0))
w_G_in_new = K_gs_new * U2 * scale * np.sqrt(ksi(P_AT, P_gs))

fig = go.Figure()

fig.add_trace(go.Surface(
    x=U2,
    y=P_AT / 1e5,
    z=w_G_in_atual,
    colorscale='Blues',
    opacity=0.75,
    name='Old model',
    showscale=False
))

fig.add_trace(go.Surface(
    x=U2,
    y=P_AT / 1e5,
    z=w_G_in_new,
    colorscale='Reds',
    opacity=0.75,
    name='New model',
    showscale=False
))

fig.update_layout(
    title='Comparison: Old vs New Valve Law',
    scene=dict(
        xaxis_title='u2',
        yaxis_title='P_at [bar]',
        zaxis_title='w_G_in'
    ),
    width=950,
    height=750
)

fig.show()