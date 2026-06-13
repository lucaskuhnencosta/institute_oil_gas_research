import torch
import math
import numpy as np
from configuration.parameters import get_parameters
parameters = get_parameters()


import torch
import numpy as np

from configuration.parameters import get_parameters

parameters = get_parameters()

Z_DIAG_NAMES = [
    # Pressures
    "P_bh_bar",
    "P_tb_b_bar",
    "P_tb_t_bar",
    "P_an_t_bar",
    "P_an_b_bar",

    # Pressure drops
    "dP_gs_an_bar",
    "dP_an_tb_bar",
    "dP_res_bh_bar",
    "dP_tb_choke_bar",

    # Main flow terms used in dx
    "w_G_in",
    "w_G_inj",
    "w_res",
    "w_L_res",
    "w_G_res",
    "w_out",
    "w_G_out",
    "w_L_out",

    # Phase/mixture variables suspected to cause blow-ups
    "alpha_L_tb_t_raw",
    "alpha_avg_L_tb",
    "alpha_L_tb_b",
    "alpha_L_tb_t",
    "alpha_G_tb_t",
    "rho_G_tb_t",
    "rho_G_tb_b",
    "rho_mix_tb_t",
    "rho_mix_tb_t_safe",
    "rho_avg_mix_tb",

    # Volumes / denominators
    "V_gas_tb_t",
    "V_gas_tb_t_safe",
    "denom_alpha_t",
    "denom_alpha_t_safe",
    "denom_alpha_b",
    "denom_alpha_b_safe",

    # Velocities / friction diagnostics
    "U_avg_mix_tb",
    "Re_tb",
    "F_t_bar",
    "F_bh_bar",
]

def softplus_stable_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Torch replica of the CasADi function:

        ax = sqrt(x*x + eps)
        softplus_stable(x) = 0.5*(x + ax) + log(1 + exp(-ax))
    """
    ax = torch.sqrt(x * x + eps)
    return 0.5 * (x + ax) + torch.log1p(torch.exp(-ax))


def smooth_pos_scaled_torch(
    dp_pa: torch.Tensor,
    scale: float = 1.0,
    k_pos: float = 20.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Torch replica of the CasADi function:

        x = (k_pos * dp_pa) / scale
        smooth_pos_scaled = (scale / k_pos) * softplus_stable(x)
    """
    x = (k_pos * dp_pa) / scale
    return (scale / k_pos) * softplus_stable_torch(x, eps=eps)


def smooth_max_scaled_torch(
    z: torch.Tensor,
    zmin: float,
    scale: float = 1.0,
    k_pos: float = 20.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Torch replica of the CasADi function:

        smooth_max_scaled(z, zmin) = zmin + smooth_pos_scaled(z - zmin)
    """
    return zmin + smooth_pos_scaled_torch(
        z - zmin,
        scale=scale,
        k_pos=k_pos,
        eps=eps,
    )


def as_torch_scalar(value, ref: torch.Tensor):
    """
    Converts a Python/NumPy scalar to a torch scalar with same dtype/device as ref.
    """
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def glc_surrogate_dx_torch(
    y: torch.Tensor,
    u: torch.Tensor,
    BSW: float,
    GOR: float,
    PI: float,
    K_gs: float,
    K_inj: float,
    K_pr: float,
    eps: float = 1e-12,
    k_pos: float = 20.0,
    return_z:bool = False,
) -> torch.Tensor:
    """
    PyTorch replica of the CasADi surrogate residual model.

    Inputs
    ------
    y : torch.Tensor, shape (..., 3)
        State vector:
            y[..., 0] = m_G_an
            y[..., 1] = m_G_tb
            y[..., 2] = m_L_tb

    u : torch.Tensor, shape (..., 2)
        Control vector:
            u[..., 0] = u1
            u[..., 1] = u2

    Returns
    -------
    dx : torch.Tensor, shape (..., 3)
        Residual vector:
            dx1 = w_G_in - w_G_inj
            dx2 = w_G_inj + w_G_res - w_G_out
            dx3 = w_L_res - w_L_out
    """

    # ------------------------------------------------
    # 0. Constants
    # ------------------------------------------------
    ref = y

    w_avg_res = as_torch_scalar(22.0, ref)

    V_an = as_torch_scalar(parameters["V_an"], ref)
    L_an = as_torch_scalar(parameters["L_an"], ref)

    D_bh = as_torch_scalar(parameters["D_bh"], ref)
    L_bh = as_torch_scalar(parameters["L_bh"], ref)
    S_bh = as_torch_scalar(parameters["A_bh"], ref)
    V_bh = as_torch_scalar(parameters["V_bh"], ref)

    L_tb = as_torch_scalar(parameters["L_tb"], ref)
    D_tb = as_torch_scalar(parameters["D_tb"], ref)
    V_tb = as_torch_scalar(parameters["V_tb"], ref)

    R = as_torch_scalar(parameters["R"], ref)
    g = as_torch_scalar(parameters["g"], ref)

    mu_o = as_torch_scalar(parameters["mu_o"], ref)
    mu_w = as_torch_scalar(parameters["mu_w"], ref)
    rho_o = as_torch_scalar(parameters["rho_o"], ref)
    rho_w = as_torch_scalar(parameters["rho_w"], ref)

    T_an = as_torch_scalar(parameters["T_an"], ref)
    T_tb = as_torch_scalar(parameters["T_tb"], ref)
    M_G = as_torch_scalar(parameters["M_G"], ref)

    P_gs = as_torch_scalar(parameters["P_gs"], ref)
    P_res = as_torch_scalar(parameters["P_res"], ref)
    P_0 = as_torch_scalar(parameters["P_0"], ref)

    epsilon_tubing = as_torch_scalar(parameters["epsilon_tubing"], ref)

    BSW_t = as_torch_scalar(BSW, ref)
    GOR_t = as_torch_scalar(GOR, ref)
    PI_t = as_torch_scalar(PI, ref)
    K_gs_t = as_torch_scalar(K_gs, ref)
    K_inj_t = as_torch_scalar(K_inj, ref)
    K_pr_t = as_torch_scalar(K_pr, ref)

    pi_t = as_torch_scalar(np.pi, ref)

    # ------------------------------------------------
    # 1. Fluid properties
    # ------------------------------------------------
    rho_L = 1.0 / (BSW_t / rho_w + (1.0 - BSW_t) / rho_o)

    mu = torch.exp(
        (1.0 - BSW_t) * torch.log(mu_o)
        + BSW_t * torch.log(mu_w)
    )

    # ------------------------------------------------
    # 2. Unpack states and controls
    # ------------------------------------------------
    m_G_an = y[..., 0]
    m_G_tb = y[..., 1]
    m_L_tb = y[..., 2]

    u1 = u[..., 0]
    u2 = u[..., 1]

    # ------------------------------------------------
    # 3. Annulus pressure
    # ------------------------------------------------
    P_an_t = R * T_an * m_G_an / (M_G * V_an)
    P_an_b = P_an_t + (m_G_an * g * L_an / V_an)

    rho_G_an_b = P_an_b * M_G / (R * T_an)
    rho_G_in = P_gs * M_G / (R * T_an)

    # ------------------------------------------------
    # 4. Gas inflow into annulus
    # ------------------------------------------------
    dP_gs_an = P_gs - P_an_t

    w_G_in = (
        K_gs_t
        * u2
        * torch.sqrt(
            rho_G_in
            * smooth_pos_scaled_torch(
                dP_gs_an,
                scale=1.0,
                k_pos=k_pos,
                eps=eps,
            )
            + eps
        )
    )

    # ------------------------------------------------
    # 5. Tubing top gas volume and pressure
    # ------------------------------------------------
    V_gas_tb_t = (V_tb + V_bh) - (m_L_tb / rho_L)
    V_gas_tb_t_safe = smooth_max_scaled_torch(
        V_gas_tb_t,
        zmin=1e-6,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    rho_G_tb_t = m_G_tb / V_gas_tb_t_safe
    P_tb_t = rho_G_tb_t * R * T_tb / M_G

    # ------------------------------------------------
    # 6. Mixture properties
    # ------------------------------------------------
    rho_avg_mix_tb = (
        m_G_tb + m_L_tb - rho_L * S_bh * L_bh
    ) / V_tb

    alpha_avg_L_tb = (
        m_L_tb - rho_L * S_bh * L_bh
    ) / (V_tb * rho_L)

    alpha_G_tb_b = GOR_t / (GOR_t + 1.0)

    U_avg_L_tb = (
        4.0
        * (1.0 - alpha_G_tb_b)
        * w_avg_res
        / (rho_L * pi_t * D_tb ** 2)
    )

    denom_G = rho_G_tb_t * pi_t * D_tb ** 2
    denom_G_safe = smooth_max_scaled_torch(
        denom_G,
        zmin=1e-8,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    U_avg_G_tb = (
        4.0
        * (w_G_in + alpha_G_tb_b * w_avg_res)
        / denom_G_safe
    )

    U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb

    # ------------------------------------------------
    # 7. Tubing Reynolds number and friction factor
    # ------------------------------------------------
    Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu

    Re_tb_safe = smooth_max_scaled_torch(
        Re_tb,
        zmin=1.0,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    log_arg_tb = (
        (epsilon_tubing / (D_tb * 3.7)) ** 1.11
        + 6.9 / Re_tb_safe
    )

    log_arg_tb_safe = torch.maximum(
        log_arg_tb,
        as_torch_scalar(1e-12, ref),
    )

    lambda_tb = (
        1.0 / (1.8 * torch.log10(log_arg_tb_safe))
    ) ** 2


    F_t = (
        alpha_avg_L_tb
        * lambda_tb
        * rho_avg_mix_tb
        * U_avg_mix_tb ** 2
        * L_tb
    ) / (2.0 * D_tb)

    P_tb_b = P_tb_t + rho_avg_mix_tb * g * L_tb + F_t

    # ------------------------------------------------
    # 8. Gas injection into tubing
    # ------------------------------------------------
    dP_an_tb = P_an_b - P_tb_b

    w_G_inj = (
        K_inj_t
        * torch.sqrt(
            rho_G_an_b
            * smooth_pos_scaled_torch(
                dP_an_tb,
                scale=1.0,
                k_pos=k_pos,
                eps=eps,
            )
            + eps
        )
    )

    # ------------------------------------------------
    # 9. Bottom-hole pressure and reservoir inflow
    # ------------------------------------------------
    U_avg_L_bh = w_avg_res / (rho_L * S_bh)

    Re_bh = rho_L * U_avg_L_bh * D_bh / mu

    log_arg_bh = (
        (epsilon_tubing / (D_bh * 3.7)) ** 1.11
        + 6.9 / Re_bh
    )

    lambda_bh = (
        1.0 / (1.8 * torch.log10(log_arg_bh))
    ) ** 2

    F_bh = (
        lambda_bh
        * rho_L
        * U_avg_L_bh ** 2
        * L_bh
    ) / (2.0 * D_bh)

    P_bh = P_tb_b + F_bh + rho_L * g * L_bh

    dP_res_bh = P_res - P_bh

    w_res = PI_t * smooth_pos_scaled_torch(
        dP_res_bh,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    w_L_res = (1.0 - alpha_G_tb_b) * w_res
    w_G_res = alpha_G_tb_b * w_res

    # ------------------------------------------------
    # 10. Bottom and top liquid fractions
    # ------------------------------------------------
    rho_G_tb_b = P_tb_b * M_G / (R * T_tb)

    denom_alpha_b = (
        w_L_res * rho_G_tb_b
        + (w_G_inj + w_G_res) * rho_L
    )

    denom_alpha_b_safe = smooth_max_scaled_torch(
        denom_alpha_b,
        zmin=1e-9,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    alpha_L_tb_b = (
        w_L_res * rho_G_tb_b
    ) / denom_alpha_b_safe

    alpha_L_tb_t_raw = 2.0 * alpha_avg_L_tb - alpha_L_tb_b
    alpha_L_tb_t = torch.clamp(alpha_L_tb_t_raw, min=1e-6, max=1.0 - 1e-6)


    rho_mix_tb_t = (
        alpha_L_tb_t * rho_L
        + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    )

    rho_mix_tb_t_safe = smooth_max_scaled_torch(
        rho_mix_tb_t,
        zmin=1e-9,
        scale=1.0,
        k_pos=k_pos,
        eps=eps,
    )

    # ------------------------------------------------
    # 11. Production choke
    # ------------------------------------------------
    dP_tb_choke = P_tb_t - P_0

    w_out = (
        K_pr_t
        * u1
        * torch.sqrt(
            rho_mix_tb_t_safe
            * smooth_pos_scaled_torch(
                dP_tb_choke,
                scale=1.0,
                k_pos=k_pos,
                eps=eps,
            )
            + eps
        )
    )

    # denom_alpha_t = (
    #     alpha_L_tb_t * rho_L
    #     + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    # )
    #
    # denom_alpha_t_safe = torch.maximum(
    #     denom_alpha_t,
    #     as_torch_scalar(1e-12, ref),
    # )
    #
    # alpha_G_tb_t = (
    #     (1.0 - alpha_L_tb_t)
    #     * rho_G_tb_t
    # ) / denom_alpha_t_safe
    #

    #
    # w_G_out = alpha_G_tb_t * w_out
    # w_L_out = (1.0 - alpha_G_tb_t) * w_out]



    denom_alpha_t = (
            alpha_L_tb_t * rho_L
            + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    )

    denom_alpha_t_safe = torch.clamp(denom_alpha_t, min=1e-9)

    alpha_G_tb_t = (
                           (1.0 - alpha_L_tb_t)
                           * rho_G_tb_t
                   ) / denom_alpha_t_safe

    alpha_G_tb_t = torch.clamp(alpha_G_tb_t, min=1e-6, max=1.0 - 1e-6)

    w_G_out = alpha_G_tb_t * w_out
    w_L_out = (1.0 - alpha_G_tb_t) * w_out

    # ------------------------------------------------
    # 12. Residuals
    # ------------------------------------------------
    dx1 = w_G_in - w_G_inj
    dx2 = w_G_inj + w_G_res - w_G_out
    dx3 = w_L_res - w_L_out

    P_bh_bar = P_bh / 1e5
    P_tb_b_bar = P_tb_b / 1e5
    P_tb_t_bar = P_tb_t / 1e5
    P_an_t_bar = P_an_t / 1e5
    P_an_b_bar = P_an_b / 1e5

    dP_gs_an_bar = dP_gs_an / 1e5
    dP_an_tb_bar = dP_an_tb / 1e5
    dP_res_bh_bar = dP_res_bh / 1e5
    dP_tb_choke_bar = dP_tb_choke / 1e5

    F_t_bar = F_t / 1e5
    F_bh_bar = F_bh / 1e5

    def bcast(x):
        """
        Broadcast scalar diagnostics to the same shape as m_G_an.
        Needed because torch.stack requires all tensors to have equal shape.
        """
        return x + torch.zeros_like(m_G_an)

    z = torch.stack(
        [
            # Pressures
            bcast(P_bh_bar),
            bcast(P_tb_b_bar),
            bcast(P_tb_t_bar),
            bcast(P_an_t_bar),
            bcast(P_an_b_bar),

            # Pressure drops
            bcast(dP_gs_an_bar),
            bcast(dP_an_tb_bar),
            bcast(dP_res_bh_bar),
            bcast(dP_tb_choke_bar),

            # Main flow terms used in dx
            bcast(w_G_in),
            bcast(w_G_inj),
            bcast(w_res),
            bcast(w_L_res),
            bcast(w_G_res),
            bcast(w_out),
            bcast(w_G_out),
            bcast(w_L_out),

            # Phase/mixture variables
            bcast(alpha_L_tb_t_raw),
            bcast(alpha_avg_L_tb),
            bcast(alpha_L_tb_b),
            bcast(alpha_L_tb_t),
            bcast(alpha_G_tb_t),
            bcast(rho_G_tb_t),
            bcast(rho_G_tb_b),
            bcast(rho_mix_tb_t),
            bcast(rho_mix_tb_t_safe),
            bcast(rho_avg_mix_tb),

            # Volumes / denominators
            bcast(V_gas_tb_t),
            bcast(V_gas_tb_t_safe),
            bcast(denom_alpha_t),
            bcast(denom_alpha_t_safe),
            bcast(denom_alpha_b),
            bcast(denom_alpha_b_safe),

            # Velocities / friction diagnostics
            bcast(U_avg_mix_tb),
            bcast(Re_tb),
            bcast(F_t_bar),
            bcast(F_bh_bar),
        ],
        dim=-1,
    )

    dx = torch.stack(
        [
            dx1,
            dx2,
            dx3,
        ],
        dim=-1,
    )

    if return_z:
        return dx, z

    return dx

    # dx = torch.stack([dx1, dx2, dx3], dim=-1)
    #
    # if return_z:
    #     return dx, z
    #
    # return dx
    #
    # dx = torch.stack(
    #     [dx1, dx2, dx3],
    #     dim=-1,
    # )
    #
    # return dx

# def smooth_relu(x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
#     """
#     Smooth approximation of max(x,0) with better gradient near 0 than
#     a very sharp softplus:
#         0.5*(x + sqrt(x^2 + delta^2))
#     """
#     d = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
#     return 0.5 * (x + torch.sqrt(x * x + d * d))
#
#
# def smooth_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     sqrt(x + eps) with clamp to keep gradient stable if x becomes tiny/negative.
#     """
#     return torch.sqrt(torch.clamp(x, min=0.0) + eps)
#
#
# def clamp_min(x: torch.Tensor, v: float) -> torch.Tensor:
#     return torch.clamp(x, min=torch.as_tensor(v, dtype=x.dtype, device=x.device))
#
#
# def glc_surrogate_dx_torch(
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
#     # Well properties ###
#     w_avg_res = 22  # kg/s is an average flow from reservoir to compute the friction terms and model bottomhole pressure to calculate the actual flow from reservoir
#
#     m_G_an = y[..., 0]
#     m_G_tb = y[..., 1]
#     m_L_tb = y[..., 2]
#     u1 = u[..., 0]
#     u2 = u[..., 1]
#
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
#     rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
#     mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
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
# #     U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb
# #
#     Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu
#     Re_tb_clamped = torch.clamp(Re_tb, 1.0, 1500000.0)

    # polynomial friction approx (as you had)


#     Re_tb_safe = clamp_min(Re_tb, 1.0)
#
#     log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
#     log_arg_tb_safe = clamp_min(log_arg_tb, 1e-12)
#
#     lambda_tb = (1.0 / (1.8 * torch.log10(log_arg_tb_safe))) ** 2
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
#
#
#     return torch.stack((dx1, dx2, dx3), dim=-1)
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