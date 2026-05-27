import torch
import math
import numpy as np
from configuration.parameters import get_parameters
parameters = get_parameters()


def smooth_relu(x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
    """
    Smooth approximation of max(x,0) with better gradient near 0 than
    a very sharp softplus:
        0.5*(x + sqrt(x^2 + delta^2))
    """
    d = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
    return 0.5 * (x + torch.sqrt(x * x + d * d))


def smooth_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    sqrt(x + eps) with clamp to keep gradient stable if x becomes tiny/negative.
    """
    return torch.sqrt(torch.clamp(x, min=0.0) + eps)


def clamp_min(x: torch.Tensor, v: float) -> torch.Tensor:
    return torch.clamp(x, min=torch.as_tensor(v, dtype=x.dtype, device=x.device))


def glc_surrogate_dx_torch(
    y: torch.Tensor,
    u: torch.Tensor,
    BSW: float,
    GOR: float,
    PI: float ,
    K_gs: float,
    K_inj: float,
    K_pr: float,
    delta_pos: float = 1e-2,   # smoother than your k_pos=20 softplus; tune 1e-3..1e-1
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Same physics structure as your glc_surrogate_dx_torch, but with
    guards chosen to avoid flat gradients / plateaus.

    - Uses smooth_relu with soft-abs (good gradients near 0)
    - Uses hard clamps for denominators/volumes/log arguments
    - Keeps everything torch-differentiable enough for Adam/LBFGS
    """

    # Well properties ###
    w_avg_res = 18  # kg/s is an average flow from reservoir to compute the friction terms and model bottomhole pressure to calculate the actual flow from reservoir

    m_G_an = y[..., 0]
    m_G_tb = y[..., 1]
    m_L_tb = y[..., 2]
    u1 = u[..., 0]
    u2 = u[..., 1]

    # Annulus
    V_an = parameters["V_an"]
    L_an = parameters["L_an"]

    # Bottom section
    D_bh = parameters["D_bh"]
    L_bh = parameters["L_bh"]
    S_bh = parameters["A_bh"]
    V_bh = parameters["V_bh"]

    # Tubing
    L_tb = parameters["L_tb"]
    D_tb = parameters["D_tb"]
    V_tb = parameters["V_tb"]

    R = parameters["R"]
    g = parameters["g"]

    mu_o = parameters["mu_o"]
    mu_w = parameters["mu_w"]
    rho_o = parameters["rho_o"]
    rho_w = parameters["rho_w"]

    T_an = parameters["T_an"]
    T_tb = parameters["T_tb"]
    M_G = parameters["M_G"]

    # Pressures
    P_gs = parameters["P_gs"]
    P_res = parameters["P_res"]
    P_0 = parameters["P_0"]

    # Friction
    epsilon_tubing = parameters["epsilon_tubing"]

    rho_L = BSW * rho_w + (1.0 - BSW) * rho_o
    mu = math.exp((1.0 - BSW) * math.log(mu_o) + BSW * math.log(mu_w))

    # ---------- Annulus pressures ----------
    P_an_t = R * T_an * m_G_an / (M_G * V_an)
    P_an_b = P_an_t + (m_G_an * g * L_an / V_an)

    rho_G_an_b = P_an_b * M_G / (R * T_an)
    rho_G_in = P_gs * M_G / (R * T_an)

    # ---------- Gas inflow to annulus ----------
    dP_gs_an = P_gs - P_an_t
    pos_dP_gs_an = smooth_relu(dP_gs_an, delta=delta_pos)
    w_G_in = K_gs * u2 * smooth_sqrt(rho_G_in * pos_dP_gs_an, eps=eps)

    # ---------- Tubing top gas volume ----------
    V_gas_tb_t = (V_tb + V_bh) - (m_L_tb / rho_L)
    V_gas_tb_t_safe = clamp_min(V_gas_tb_t, 1e-6)  # hard clamp avoids flat softplus tail
    rho_G_tb_t = m_G_tb / V_gas_tb_t_safe

    P_tb_t = rho_G_tb_t * R * T_tb / M_G

    # ---------- Mixture properties ----------
    alpha_avg_L_tb = (m_L_tb - rho_L * S_bh * L_bh) / (V_tb * rho_L)
    rho_avg_mix_tb = (m_G_tb + m_L_tb - rho_L * S_bh * L_bh) / V_tb

    alpha_G_tb_b = GOR / (GOR + 1.0)

    U_avg_L_tb = 4.0 * (1.0 - alpha_G_tb_b) * w_avg_res / (rho_L * math.pi * D_tb ** 2)

    denom_G = rho_G_tb_t * math.pi * D_tb ** 2
    denom_G_safe = clamp_min(denom_G, 1e-8)
    U_avg_G_tb = 4.0 * (w_G_in + alpha_G_tb_b * w_avg_res) / denom_G_safe
    U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb

    Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu
    Re_tb_clamped = torch.clamp(Re_tb, 10000.0, 150000.0)

    # polynomial friction approx (as you had)
    lambda_tb_approx = torch.tensor(
        [-1.78223894e-17, 4.56100539e-12, -4.18248919e-07, 3.29432465e-02],
        dtype=y.dtype,
        device=y.device,
    )
    lambda_tb = (
        lambda_tb_approx[0] * Re_tb_clamped ** 3
        + lambda_tb_approx[1] * Re_tb_clamped ** 2
        + lambda_tb_approx[2] * Re_tb_clamped
        + lambda_tb_approx[3]
    )

    F_t = (alpha_avg_L_tb * lambda_tb * rho_avg_mix_tb * U_avg_mix_tb ** 2 * L_tb) / (2.0 * D_tb)
    P_tb_b = P_tb_t + rho_avg_mix_tb * g * L_tb + F_t

    # ---------- Injection ----------
    dP_an_tb = P_an_b - P_tb_b
    pos_dP_an_tb = smooth_relu(dP_an_tb, delta=delta_pos)
    w_G_inj = K_inj * smooth_sqrt(rho_G_an_b * pos_dP_an_tb, eps=eps)

    # ---------- Bottom-hole friction ----------
    U_avg_L_bh = w_avg_res / (rho_L * S_bh)
    Re_bh = rho_L * U_avg_L_bh * D_bh / mu

    # compute lambda_bh with a *hard clamped* log argument (stable)
    log_arg_bh = (epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh
    log_arg_bh_t = torch.as_tensor(log_arg_bh, dtype=y.dtype, device=y.device)
    log_arg_bh_t = clamp_min(log_arg_bh_t, 1e-12)
    lambda_bh = (1.0 / (1.8 * torch.log10(log_arg_bh_t))) ** 2

    F_bh = (lambda_bh * rho_L * U_avg_L_bh ** 2 * L_bh) / (2.0 * D_bh)
    P_bh = P_tb_b + F_bh + rho_L * g * L_bh

    # ---------- Reservoir inflow ----------
    dP_res_bh = P_res - P_bh
    pos_dP_res_bh = smooth_relu(dP_res_bh, delta=delta_pos)
    w_res = PI * pos_dP_res_bh

    w_L_res = (1.0 - alpha_G_tb_b) * w_res
    w_G_res = alpha_G_tb_b * w_res

    rho_G_tb_b = (P_tb_b * M_G) / (R * T_tb)

    denom_alpha_b = (w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * rho_L)
    denom_alpha_b_safe = clamp_min(denom_alpha_b, 1e-9)
    alpha_L_tb_b = (w_L_res * rho_G_tb_b) / denom_alpha_b_safe

    alpha_L_tb_t = 2.0 * alpha_avg_L_tb - alpha_L_tb_b

    rho_mix_tb_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    rho_mix_tb_t_safe = clamp_min(rho_mix_tb_t, 1e-9)

    # ---------- Production choke ----------
    dP_tb_choke = P_tb_t - P_0
    pos_dP_tb_choke = smooth_relu(dP_tb_choke, delta=delta_pos)
    w_out = K_pr * u1 * smooth_sqrt(rho_mix_tb_t_safe * pos_dP_tb_choke, eps=eps)

    denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    denom_alpha_t_safe = clamp_min(denom_alpha_t, 1e-12)
    alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb_t) / denom_alpha_t_safe

    w_G_out = alpha_G_tb_t * w_out
    w_L_out = (1.0 - alpha_G_tb_t) * w_out

    dx1 = w_G_in - w_G_inj
    dx2 = w_G_inj + w_G_res - w_G_out
    dx3 = w_L_res - w_L_out

    return torch.stack((dx1, dx2, dx3), dim=-1)
#
# def glc_surrogate_dx_torch_alg(
#     y: torch.Tensor,
#     u: torch.Tensor,
#     BSW: float,
#     GOR: float,
#     PI: float ,
#     K_gs: float,
#     K_inj: float,
#     K_pr: float,
#     delta_pos: float = 1e-2,   # smoother than your k_pos=20 softplus; tune 1e-3..1e-1
#     eps: float = 1e-12,
# ) -> torch.Tensor:
#     """
#     Same physics structure as your glc_surrogate_dx_torch, but with
#     guards chosen to avoid flat gradients / plateaus.
#
#     - Uses smooth_relu with soft-abs (good gradients near 0)
#     - Uses hard clamps for denominators/volumes/log arguments
#     - Keeps everything torch-differentiable enough for Adam/LBFGS
#     """
#
#     w_avg_res = 18.0
#
#     m_G_an = y[..., 0]
#     m_G_tb = y[..., 1]
#     m_L_tb = y[..., 2]
#     u1 = u[..., 0]
#     u2 = u[..., 1]
#     # Annulus
#     V_an = parameters["V_an"]
#     L_an = parameters["L_an"]
#
#     # Bottom section
#     D_bh = parameters["D_bh"]
#     L_bh = parameters["L_bh"]
#     S_bh = parameters["A_bh"]
#     V_bh = parameters["V_bh"]
#
#     # Tubing
#     L_tb = parameters["L_tb"]
#     D_tb = parameters["D_tb"]
#     V_tb = parameters["V_tb"]
#
#     R = parameters["R"]
#     g = parameters["g"]
#
#     mu_o = parameters["mu_o"]
#     mu_w = parameters["mu_w"]
#     rho_o = parameters["rho_o"]
#     rho_w = parameters["rho_w"]
#
#     T_an = parameters["T_an"]
#     T_tb = parameters["T_tb"]
#     M_G = parameters["M_G"]
#
#     # Pressures
#     P_gs = parameters["P_gs"]
#     P_res = parameters["P_res"]
#     P_0 = parameters["P_0"]
#
#     # Friction
#     epsilon_tubing = parameters["epsilon_tubing"]
#
#     rho_L = BSW * rho_w + (1.0 - BSW) * rho_o
#     mu = math.exp((1.0 - BSW) * math.log(mu_o) + BSW * math.log(mu_w))
#
#     # ---------- Annulus pressures ----------
#     P_an_t = R * T_an * m_G_an / (M_G * V_an)
#     P_an_b = P_an_t + (m_G_an * g * L_an / V_an)
#
#     rho_G_an_b = P_an_b * M_G / (R * T_an)
#     rho_G_in = P_gs * M_G / (R * T_an)
#
#     # ---------- Gas inflow to annulus ----------
#     dP_gs_an = P_gs - P_an_t
#     pos_dP_gs_an = smooth_relu(dP_gs_an, delta=delta_pos)
#     w_G_in = K_gs * u2 * smooth_sqrt(rho_G_in * pos_dP_gs_an, eps=eps)
#
#     # ---------- Tubing top gas volume ----------
#     V_gas_tb_t = (V_tb + V_bh) - (m_L_tb / rho_L)
#     V_gas_tb_t_safe = clamp_min(V_gas_tb_t, 1e-6)  # hard clamp avoids flat softplus tail
#     rho_G_tb_t = m_G_tb / V_gas_tb_t_safe
#
#     P_tb_t = rho_G_tb_t * R * T_tb / M_G
#
#     # ---------- Mixture properties ----------
#     alpha_avg_L_tb = (m_L_tb - rho_L * S_bh * L_bh) / (V_tb * rho_L)
#     rho_avg_mix_tb = (m_G_tb + m_L_tb - rho_L * S_bh * L_bh) / V_tb
#
#     alpha_G_tb_b = GOR / (GOR + 1.0)
#
#     U_avg_L_tb = 4.0 * (1.0 - alpha_G_tb_b) * w_avg_res / (rho_L * math.pi * D_tb ** 2)
#
#     denom_G = rho_G_tb_t * math.pi * D_tb ** 2
#     denom_G_safe = clamp_min(denom_G, 1e-8)
#     U_avg_G_tb = 4.0 * (w_G_in + alpha_G_tb_b * w_avg_res) / denom_G_safe
#     U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb
#
#     Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu
#     Re_tb_clamped = torch.clamp(Re_tb, 10000.0, 150000.0)
#
#     # polynomial friction approx (as you had)
#     lambda_tb_approx = torch.tensor(
#         [-1.78223894e-17, 4.56100539e-12, -4.18248919e-07, 3.29432465e-02],
#         dtype=y.dtype,
#         device=y.device,
#     )
#     lambda_tb = (
#         lambda_tb_approx[0] * Re_tb_clamped ** 3
#         + lambda_tb_approx[1] * Re_tb_clamped ** 2
#         + lambda_tb_approx[2] * Re_tb_clamped
#         + lambda_tb_approx[3]
#     )
#
#     F_t = (alpha_avg_L_tb * lambda_tb * rho_avg_mix_tb * U_avg_mix_tb ** 2 * L_tb) / (2.0 * D_tb)
#     P_tb_b = P_tb_t + rho_avg_mix_tb * g * L_tb + F_t
#
#     # ---------- Injection ----------
#     dP_an_tb = P_an_b - P_tb_b
#     pos_dP_an_tb = smooth_relu(dP_an_tb, delta=delta_pos)
#     w_G_inj = K_inj * smooth_sqrt(rho_G_an_b * pos_dP_an_tb, eps=eps)
#
#     # ---------- Bottom-hole friction ----------
#     U_avg_L_bh = w_avg_res / (rho_L * S_bh)
#     Re_bh = rho_L * U_avg_L_bh * D_bh / mu
#
#     # compute lambda_bh with a *hard clamped* log argument (stable)
#     log_arg_bh = (epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh
#     log_arg_bh_t = torch.as_tensor(log_arg_bh, dtype=y.dtype, device=y.device)
#     log_arg_bh_t = clamp_min(log_arg_bh_t, 1e-12)
#     lambda_bh = (1.0 / (1.8 * torch.log10(log_arg_bh_t))) ** 2
#
#     F_bh = (lambda_bh * rho_L * U_avg_L_bh ** 2 * L_bh) / (2.0 * D_bh)
#     P_bh = P_tb_b + F_bh + rho_L * g * L_bh
#
#     # ---------- Reservoir inflow ----------
#     dP_res_bh = P_res - P_bh
#     pos_dP_res_bh = smooth_relu(dP_res_bh, delta=delta_pos)
#     w_res = PI * pos_dP_res_bh
#
#     w_L_res = (1.0 - alpha_G_tb_b) * w_res
#     w_G_res = alpha_G_tb_b * w_res
#
#     rho_G_tb_b = (P_tb_b * M_G) / (R * T_tb)
#
#     denom_alpha_b = (w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * rho_L)
#     denom_alpha_b_safe = clamp_min(denom_alpha_b, 1e-9)
#     alpha_L_tb_b = (w_L_res * rho_G_tb_b) / denom_alpha_b_safe
#
#     alpha_L_tb_t = 2.0 * alpha_avg_L_tb - alpha_L_tb_b
#
#     rho_mix_tb_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
#     rho_mix_tb_t_safe = clamp_min(rho_mix_tb_t, 1e-9)
#
#     # ---------- Production choke ----------
#     dP_tb_choke = P_tb_t - P_0
#     pos_dP_tb_choke = smooth_relu(dP_tb_choke, delta=delta_pos)
#     w_out = K_pr * u1 * smooth_sqrt(rho_mix_tb_t_safe * pos_dP_tb_choke, eps=eps)
#
#     denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
#     denom_alpha_t_safe = clamp_min(denom_alpha_t, 1e-12)
#     alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb_t) / denom_alpha_t_safe
#
#     w_G_out = alpha_G_tb_t * w_out
#     w_L_out = (1.0 - alpha_G_tb_t) * w_out
#
#     dx1 = w_G_in - w_G_inj
#     dx2 = w_G_inj + w_G_res - w_G_out
#     dx3 = w_L_res - w_L_out
#
#     # Algebraic exits
#     P_bh_bar=P_bh/1e5
#     P_tb_b_bar=P_tb_b/1e5
#
#
#     return torch.stack((P_bh_bar, P_tb_b_bar), dim=-1)