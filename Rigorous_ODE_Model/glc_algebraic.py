import torch
import math
import numpy as np


def glc_f_alg(y,u):
    """
    This function is an exact copy of glc_f, but outputs algebraic values instead of dx
    """
    R = 8.314 #J/(K*mol) is the universal gas constant
    g = 9.81 #m/s^2 is the gravity
    mu = 3.64e-3 # Pa.s is the viscosity
    rho_L = 760 #kg/m^3 is the density of the liquid in the tubing
    M_G = 0.0167 #(kg/mol) is the gas molecular weight
    T_an = 348 #K is the annulus temperature
    V_an = 64.34  #m^3 is the annulus volume
    L_an = 2048 #m is the length of the annulus
    P_gs = 140e5  #140bar is the gas source pressure
    V_tb = 25.03 #m^3 is the volume of the tubing
    S_bh = 0.0314  #m^2 is the cross section below the injection point
    D_bh = 2 * np.sqrt(S_bh / np.pi)
    L_bh = 75  #m is the length below the injection point
    T_tb = 369.4 #K is the tubing temperature
    GOR = 0 #is the gas oil ratio
    P_res = 160e5 #160bar, the constant reservoir pressure
    w_avg_res = 18 #kg/s is an average flow from reservoir to compute the friction terms and model bottomhole pressure to calculate the actual flow from reservoir
    D_tb = 0.134 #tubing diameter
    L_tb = 2048 #m, tubing length
    PI = 2.47e-6 #is the productivity index in kg/(s.Pa)
    K_gs = 9.98e-5 #is the gas lift choke constant
    K_inj = 1.40e-4 #is the injection valve choke constant
    K_pr = 2.90e-3 #is the production choke constant
    epsilon_tubing = 2.8e-5   #[m] piping superficial roughness, taken from Jeans' dissertation
    P_0 = 20e5  # pressure downstream of choke
    m_G_an = y[..., 0]
    m_G_tb = y[..., 1]
    m_L_tb = y[..., 2]
    u1 = u[..., 0]
    u2 = u[..., 1]


    # Equation 1 - Annulus top pressure using ideal gas law
    P_an_t=R*T_an*m_G_an/(M_G*V_an)

    # Equation 2 - Annulus bottom pressure
    P_an_b=P_an_t+(m_G_an*g*L_an/V_an)

    # Equation 3 - Density of gas phase at annulus bottom using ideal gas law
    rho_G_an_b=P_an_b*M_G/(R*T_an)

    # Equation 4 - Density of gas through gas-lift choke using ideal gas law
    rho_G_in=P_gs*M_G/(R*T_an)

    # Equation 5 - Mass flow into annulus using Bernoulli's equation for orifice flow
    w_G_in =K_gs*u2*torch.sqrt(rho_G_in*torch.maximum(P_gs-P_an_t,torch.tensor(0.0, device=y.device)))

    # Equation 6 - Density of the gas at top of the tubing
    rho_G_tb_t=m_G_tb/(V_tb+S_bh*L_bh-m_L_tb/rho_L)

    # Equation 7 - Pressure at top of tubing using ideal gas law
    P_tb_t=rho_G_tb_t*R*T_tb/M_G

    # Equation 8 - Average mixture density inside tubing (above injection point)
    rho_avg_mix_tb=(m_G_tb+m_L_tb-rho_L*S_bh*L_bh)/V_tb

    # Equation 9 - Average liquid volume fraction inside tubing (above injection point)
    alpha_avg_L_tb=(m_L_tb-rho_L*S_bh*L_bh)/(V_tb*rho_L)

    # Equation 10 - Gas mass fraction at bottom-hole, as the GOR is constant this is also constant and not an algebraic equation
    alpha_G_tb_b = GOR / (GOR + 1)

    # Equation 11 - Average superficial velocity of liquid phase in tubing
    U_avg_L_tb=(4*(1-alpha_G_tb_b)*w_avg_res/(rho_L*np.pi*D_tb**2))

    # Equation 12 - Average superficial velocity of gas phase
    U_avg_G_tb=(4*(w_G_in+alpha_G_tb_b*w_avg_res)/torch.maximum(rho_G_tb_t*np.pi*D_tb**2,torch.tensor(1e-5,device=y.device)))

    # Equation 13 - Average mixture velocity in tubing
    U_avg_mix_tb = (U_avg_L_tb + U_avg_G_tb)

    # Equation 14 - Reynolds number of flow in tubing
    Re_tb=rho_avg_mix_tb*U_avg_mix_tb*D_tb/mu

    # Equation 15 - Friction factor in tubing
    log_arg_tb = torch.tensor((epsilon_tubing / (D_tb * 3.7)) ** 1.11, device=y.device) + 6.9 / Re_tb
    lambda_tb = (1 / (1.8 * torch.log10(log_arg_tb))) ** 2

    # Equation 16 - Pressure loss due to friction in tubing
    F_t=(alpha_avg_L_tb*lambda_tb*rho_avg_mix_tb*U_avg_mix_tb**2*L_tb)/(2*D_tb)

    # Equation 17 - Pressure at tubing bottom
    P_tb_b=P_tb_t+rho_avg_mix_tb*g*L_tb+F_t

    # Equation 18 - Mass flow rate of gas injected into tubing
    w_G_inj=K_inj*torch.sqrt(rho_G_an_b*torch.maximum(P_an_b-P_tb_b,torch.tensor(0.0,device=y.device)))

    # Equation 19 - Liquid velocity at bottom-hole
    U_avg_L_bh=w_avg_res/(rho_L*S_bh)

    # Equation 20 - Reynolds number of flow at bottom-hole
    Re_bh=rho_L*U_avg_L_bh*D_bh/mu

    # Equation 21 - Friction factor at bottom hole
    log_arg_bh = torch.tensor((epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh, device=y.device)
    lambda_bh = (1 / (1.8 * torch.log10(log_arg_bh))) ** 2

    # Equation 22 - Pressure loss due to friction from bottom-hole to injection point
    F_bh=(lambda_bh*rho_L*U_avg_L_bh**2*L_bh)/(2*D_bh)

    # Equation 23 - Pressure at bottom-hole
    P_bh=P_tb_b+F_bh+rho_L*g*L_bh

    # Equation 24 - Mass flow rate from reservoir to tubing
    w_res = PI * torch.maximum(P_res - P_bh, torch.tensor(0.0, device=y.device))

    # Equation 25 - Mass flow rate of liquid from reservoir to tubing
    w_L_res=(1-alpha_G_tb_b)*w_res

    # Equation 26 - Mass flow rate of gas from reservoir to the well
    w_G_res=alpha_G_tb_b*w_res

    # Equation 27 - Density of gas at bottom of tubing
    rho_G_tb_b=(P_tb_b*M_G)/(R*T_tb)

    # Equation 28 - Liquid volume fraction at bottom of tubing
    alpha_L_tb_b=(w_L_res*rho_G_tb_b)/(w_L_res*rho_G_tb_b+(w_G_inj+w_G_res)*rho_L)

    # Equation 29 - Liquid volume fraction at top of the tubing
    alpha_L_tb_t=2*alpha_avg_L_tb-alpha_L_tb_b

    # Equation 30 - Mixture density at the top
    rho_mix_tb_t=alpha_L_tb_t*rho_L+(1-alpha_L_tb_t)*rho_G_tb_t

    # Equation 31 - Mass flow rate of mixture from choke
    w_out = K_pr * u1 * torch.sqrt(rho_mix_tb_t * torch.maximum(P_tb_t - P_0, torch.tensor(0.0, device=y.device)))

    # Equation 32 - Volumetric flow rate of production choke
    Q_out=w_out/rho_mix_tb_t

    # Equation 33 - Gas Mass Fraction at top of tubing
    alpha_G_tb_t=((1-alpha_L_tb_t)*rho_G_tb_t)/(alpha_L_tb_t*rho_L+(1-alpha_L_tb_t)*rho_G_tb_t)

    # Equation 34 - Mass flow rate of outlet gas
    w_G_out=alpha_G_tb_t*w_out

    # Equation 35 - Mass flow rate of outlet liquid
    w_L_out=(1-alpha_G_tb_t)*w_out

    dx1 = w_G_in - w_G_inj              # (change of) mass of gas in annulus
    dx2 = w_G_inj + w_G_res - w_G_out   # (change of) mass of gas in tubing
    dx3 = w_L_res - w_L_out             # (change of) mass of liquid in tubing

    P_bh_bar = P_bh / 1e5

    print(f"alpha is: {alpha_L_tb_b} {alpha_G_tb_b} and {alpha_L_tb_t}")
    print(f"desity of gas in the tubing is: {rho_G_tb_t}, pressure is: {P_tb_t}")
    print(f"desnity of the mix at tubing top is: {rho_mix_tb_t}")
    print(f"Reynolds number is: {Re_tb}, {log_arg_tb}, {lambda_tb}")

    return torch.stack((P_bh_bar,
                        w_L_out,  # Proxy for oil + water production
                        w_G_inj,  # Injected gas into tubing
                        w_G_in,  # Gas supplied to annulus
                        w_G_out,  # Proxy for gas production
                        P_tb_t/1e5), dim=1)