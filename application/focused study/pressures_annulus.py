from casadi import *
import numpy as np
import casadi as ca


k_pos = 20
eps = 1e-12

# # Well properties ###
BSW = 0
GOR = 0 # is the gas oil ratio
PI = 3.00e-6  # is the productivity index in kg/(s.Pa)

# Geometry and temperature of the wellbore
### Annulus ###
V_an = 64.34  # m^3 is the annulus volume
L_an = 2048  # m is the length of the annulus
T_an = 348  # K is the annulus temperature

### Tubing bottom ###
D_bh=0.2
L_bh=75
S_bh=(np.pi*D_bh**2)/4
V_bh = S_bh * L_bh
T_bh = 371.5

### Tubing ###
L_tb = 1973
D_tb = 0.134
S_tb = (np.pi*D_tb**2)/4
V_tb = S_tb * L_tb
T_tb = 369.4  # K is the tubing temperature

V_tb=V_tb-V_bh

# Constants (general)
R = 8.314  # J/(K*mol) is the universal gas constant
g = 9.81  # m/s^2 is the gravity
mu_o = 3.64e-3  # Pa.s is the viscosity
mu_w = 1.00e-3
rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
rho_w = 1000
rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
M_G = 0.0167  # (kg/mol) is the gas molecular weight

# Pressures
P_gs = 140e5  # 140bar is the gas source pressure
P_res = 160e5  # 160bar, the constant reservoir pressure
P_0 = 20e5  # pressure downstream of choke

# Chokes
K_gs = 9.98e-5  # is the gas lift choke constant
# K_gs=1e-4
K_inj = 1.40e-4  # is the injection valve choke constant
K_pr = 2.90e-3  # is the production choke constant
K0_int=2.50

# Friction
epsilon_tubing = 1e-3

# ---------- unpack ----------
# m_G_an=y[0]
# m_G_t=y[1]
# m_o_t=y[2]
# m_w_t=y[3]
# m_G_b=y[4]
# m_o_b=y[5]
# m_w_b =y[6]
# u1=u[0]
# u2=u[1]

# Algebraic variables
#
# P_bh_g=z[0] #Pressure bottomhole, closed at g6
# P_tb_b_g=z[1] # Pressure at the bottom of the tubing, closed at g7
# w_res_g=z[2] # Reservoir flow in kg/s
# w_up_g=z[3] # Flow into the tubing in kg/s


# -----------------------
# PART 1 - RESERVOIR INFLOW
# -----------------------
m_G_an = np.linspace(3200.0, 4800.0, 300)
# w_L_res=w_res_g/(1+GOR) # 1.1
# w_o_res=(1-BSW)*w_L_res # 1.2
# w_w_res=BSW*w_L_res  #1.3
# w_G_res=GOR*w_L_res #1.4


P_an_t_old=R*T_an*m_G_an/(M_G*V_an) # 2.1
P_an_b_old=P_an_t_old+(m_G_an*g*L_an/V_an) # 2.2

P_an_t_new=(m_G_an*g/(V_an/L_an))*(
    np.exp(-g*M_G*L_an/(R*T_an)) /(1-np.exp(-g*M_G/(R*T_an)*L_an)))

P_an_b_new=(m_G_an*g/(V_an/L_an))*(
        1/(1-np.exp(-g*M_G/(R*T_an)*L_an)))

import numpy as np
import matplotlib.pyplot as plt
# -----------------------------

# -----------------------------
# Convert to bar
# -----------------------------
P_an_t_old_bar = P_an_t_old / 1e5
P_an_b_old_bar = P_an_b_old / 1e5
P_an_t_new_bar = P_an_t_new / 1e5
P_an_b_new_bar = P_an_b_new / 1e5

# -----------------------------
# Plot 1: all 4 curves together
# -----------------------------
plt.figure(figsize=(9, 6))
plt.plot(m_G_an, P_an_t_old_bar, label='Old $P_{an,t}$', linewidth=2)
plt.plot(m_G_an, P_an_b_old_bar, label='Old $P_{an,b}$', linewidth=2)
plt.plot(m_G_an, P_an_t_new_bar, label='New $P_{an,t}$', linewidth=2, linestyle='--')
plt.plot(m_G_an, P_an_b_new_bar, label='New $P_{an,b}$', linewidth=2, linestyle='--')

plt.xlabel(r'$m_{G,an}$ [kg]')
plt.ylabel('Pressure [bar]')
plt.title('Annulus pressure comparison vs gas mass')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: difference old vs new
# -----------------------------
plt.figure(figsize=(9, 6))
plt.plot(m_G_an, P_an_t_new_bar, label='new top', linewidth=2)
plt.plot(m_G_an,P_an_t_old_bar, label='old top', linewidth=2)
plt.plot(m_G_an, P_an_b_new_bar, label='new bottom', linewidth=2)
plt.plot(m_G_an, P_an_b_old_bar, label='old bottom', linewidth=2)
plt.xlabel(r'$m_{G,an}$ [kg]')
plt.ylabel('Pressure difference [bar]')
plt.title('Difference between new and old annulus pressure models')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()