import torch
import math
import numpy as np

def softplus_stable(x: torch.Tensor,
                    eps: float = 1e-12) -> torch.Tensor:
    ax = torch.sqrt(x * x + eps)
    return 0.5 * (x + ax) + torch.log1p(torch.exp(-ax))

def smooth_pos_scaled(dp: torch.Tensor,
                      k_pos: float = 20.0,
                      scale: float = 1.0) -> torch.Tensor:
    x = (k_pos * dp) / scale
    return (scale / k_pos) * softplus_stable(x)

def smooth_max_scaled(z: torch.Tensor,
                           zmin: float,
                           k_pos: float = 20.0,
                           scale: float = 1.0) -> torch.Tensor:
    zmin_t = torch.as_tensor(zmin, dtype=z.dtype, device=z.device)
    return zmin_t + smooth_pos_scaled(z - zmin_t, k_pos=k_pos, scale=scale)

def glc_surrogate_dx_torch(y: torch.Tensor,
                           u: torch.Tensor,
                           BSW: float=0.20,
                           GOR: float=0.05,
                           PI: float=3.0e-6,
                           k_pos: float=20.0,
                           eps: float=1e-12) -> torch.Tensor:

    w_avg_res = 18

    m_G_an = y[..., 0]
    m_G_tb = y[..., 1]
    m_L_tb = y[..., 2]
    u1 = u[..., 0]
    u2 = u[..., 1]

    # Geometry and temperature of the wellbore
    ### Annulus ###
    V_an = 64.34  # m^3 is the annulus volume
    L_an = 2048  # m is the length of the annulus
    T_an = 348  # K is the annulus temperature

    ### Tubing bottom ###
    D_bh = 0.2
    L_bh = 75
    S_bh = (np.pi * D_bh ** 2) / 4
    V_bh = S_bh * L_bh

    ### Tubing ###
    L_tb = 1973
    D_tb = 0.134
    S_tb = (np.pi * D_tb ** 2) / 4
    V_tb = S_tb * L_tb
    T_tb = 369.4  # K is the tubing temperature

    # Constants (general)
    R = 8.314  # J/(K*mol) is the universal gas constant
    g = 9.81  # m/s^2 is the gravity
    mu_o = 3.64e-3  # Pa.s is the viscosity
    mu_w = 1.00e-3
    rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
    rho_w = 1000
    rho_L = BSW * rho_w + (1 - BSW) * rho_o
    mu = math.exp((1 - BSW) * math.log(mu_o) + BSW * math.log(mu_w))
    M_G = 0.0167  # (kg/mol) is the gas molecular weight

    # Pressures
    P_gs = 140e5  # 140bar is the gas source pressure
    P_res = 160e5  # 160bar, the constant reservoir pressure
    P_0 = 20e5  # pressure downstream of choke

    # Chokes
    K_gs = 9.98e-5  # is the gas lift choke constant
    K_inj = 1.40e-4  # is the injection valve choke constant
    K_pr = 2.90e-3  # is the production choke constant

    # Friction
    epsilon_tubing = 1e-3

    P_an_t=R*T_an*m_G_an/(M_G*V_an)
    P_an_b=P_an_t+(m_G_an*g*L_an/V_an)

    rho_G_an_b=P_an_b*M_G/(R*T_an)
    rho_G_in=P_gs*M_G/(R*T_an)

    dP_gs_an = P_gs - P_an_t
    w_G_in =K_gs*u2*torch.sqrt(rho_G_in*smooth_pos_scaled(dP_gs_an,k_pos=k_pos,scale=1.0)+eps)

    # gas volume at tubing top derived from liquid mass
    V_gas_tb_t = (V_tb + V_bh) - (m_L_tb / rho_L)
    V_gas_tb_t_safe = smooth_max_scaled(V_gas_tb_t, 1e-6, k_pos=k_pos, scale=1.0)
    rho_G_tb_t = m_G_tb / V_gas_tb_t_safe

    P_tb_t=rho_G_tb_t*R*T_tb/M_G

    # hold-ups / mixture
    alpha_avg_L_tb = (m_L_tb - rho_L * S_bh * L_bh) / (V_tb * rho_L)
    rho_avg_mix_tb = (m_G_tb + m_L_tb - rho_L * S_bh * L_bh) / V_tb

    alpha_G_tb_b = GOR / (GOR + 1.0)

    U_avg_L_tb = 4.0 * (1.0 - alpha_G_tb_b) * w_avg_res / (rho_L * math.pi * D_tb ** 2)

    denom_G = rho_G_tb_t * math.pi * D_tb ** 2
    denom_G_safe = smooth_max_scaled(denom_G, 1e-8, k_pos=k_pos, scale=1.0)
    U_avg_G_tb = 4.0 * (w_G_in + alpha_G_tb_b * w_avg_res) / denom_G_safe

    U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb

    Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu
    Re_tb_safe = smooth_max_scaled(Re_tb, 1.0, k_pos=k_pos, scale=1.0)

    log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
    log_arg_tb_safe = smooth_max_scaled(log_arg_tb, 1e-12, k_pos=k_pos, scale=1.0)
    lambda_tb = (1.0 / (1.8 * torch.log10(log_arg_tb_safe))) ** 2

    F_t = (alpha_avg_L_tb * lambda_tb * rho_avg_mix_tb * U_avg_mix_tb ** 2 * L_tb) / (2.0 * D_tb)
    P_tb_b = P_tb_t + rho_avg_mix_tb * g * L_tb + F_t

    # injection: smooth positive
    dP_an_tb = P_an_b - P_tb_b
    w_G_inj = K_inj * torch.sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb, k_pos=k_pos, scale=1.0) + eps)

    # bottom-hole friction
    U_avg_L_bh = w_avg_res / (rho_L * S_bh)
    Re_bh = rho_L * U_avg_L_bh * D_bh / mu

    log_arg_bh = (epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh
    log_arg_bh_safe = smooth_max_scaled(torch.as_tensor(log_arg_bh, dtype=y.dtype, device=y.device), 1e-12,
                                        k_pos=k_pos, scale=1.0)
    lambda_bh = (1.0 / (1.8 * torch.log10(log_arg_bh_safe))) ** 2

    F_bh = (lambda_bh * rho_L * U_avg_L_bh ** 2 * L_bh) / (2.0 * D_bh)
    P_bh = P_tb_b + F_bh + rho_L * g * L_bh

    # reservoir inflow: smooth positive
    dP_res_bh = P_res - P_bh
    w_res = PI * smooth_pos_scaled(dP_res_bh, k_pos=k_pos, scale=1.0)

    w_L_res = (1.0 - alpha_G_tb_b) * w_res
    w_G_res = alpha_G_tb_b * w_res

    rho_G_tb_b = (P_tb_b * M_G) / (R * T_tb)

    denom_alpha_b = (w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * rho_L)
    denom_alpha_b_safe = smooth_max_scaled(denom_alpha_b, 1e-9, k_pos=k_pos, scale=1.0)
    alpha_L_tb_b = (w_L_res * rho_G_tb_b) / denom_alpha_b_safe

    alpha_L_tb_t = 2.0 * alpha_avg_L_tb - alpha_L_tb_b

    rho_mix_tb_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    rho_mix_tb_t_safe = smooth_max_scaled(rho_mix_tb_t, 1e-9, k_pos=k_pos, scale=1.0)

    dP_tb_choke = P_tb_t - P_0
    w_out = K_pr * u1 * torch.sqrt(rho_mix_tb_t_safe * smooth_pos_scaled(dP_tb_choke, k_pos=k_pos, scale=1.0) + eps)

    denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    denom_alpha_t_safe = smooth_max_scaled(denom_alpha_t, 1e-12, k_pos=k_pos, scale=1.0)
    alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb_t) / denom_alpha_t_safe

    w_G_out = alpha_G_tb_t * w_out
    w_L_out = (1.0 - alpha_G_tb_t) * w_out

    dx1 = w_G_in - w_G_inj
    dx2 = w_G_inj + w_G_res - w_G_out
    dx3 = w_L_res - w_L_out

    return torch.stack((dx1, dx2, dx3), dim=-1)