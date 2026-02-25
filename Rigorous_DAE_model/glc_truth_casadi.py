from casadi import *
import numpy as np
import casadi as ca


def glc_well_01_rigorous_casadi(y, z, u):
    """
    Complete (true friction) gas-lift well model in CasADi with smooth positivity
    for IPOPT-friendliness

    Inputs:
         y: CasADi SX/MX (3,)  [m_G_an, m_G_tb, m_L_tb]
        u: CasADi SX/MX (2,)  [u1, u2]

    Gas-lift well ODE RHS + selected algebraic outputs (single-source-of-truth).

    Inputs:
      y: CasADi MX/SX (3x1)  -> [m_G_an, m_G_tb, m_L_tb]
      u: CasADi MX/SX (2x1)  -> [u1, u2]

    Returns:
      dx: (3x1) ODE RHS
      z : (6x1) algebraic outputs:
           [P_bh_bar,
            w_L_out,
            w_G_inj,
            w_G_in,
            w_G_out,
            P_tb_t_bar]
    """

    def softplus_stable(x):
        ax = sqrt(x * x + tiny)
        return 0.5 * (x + ax) + log(1 + exp(-ax))

    def smooth_pos_scaled(dp_pa):
        x = (k_pos * dp_pa)
        return (1 / k_pos) * (softplus_stable(x))

    def smooth_max_scaled(z, zmin):
        # smooth approximation of max(z,zmin))
        return zmin + smooth_pos_scaled(z - zmin)

    k_pos = 20
    eps = 1e-12
    tiny = 1e-12

    BSW = 0
    GOR = 0 # is the gas oil ratio
    PI = 3.00e-6  # is the productivity index in kg/(s.Pa)
    #2.47
    R = 8.314  # J/(K*mol) is the universal gas constant
    g = 9.81  # m/s^2 is the gravity
    mu_o = 3.64e-3  # Pa.s is the viscosity
    mu_w = 1.00e-3
    rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
    rho_w = 1000
    rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
    mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))

    M_G = 0.0167  # (kg/mol) is the gas molecular weight
    T_an = 348  # K is the annulus temperature
    V_an = 64.34  # m^3 is the annulus volume
    L_an = 2048  # m is the length of the annulus
    P_gs = 140e5  # 140bar is the gas source pressure

    V_tb = 25.03  # m^3 is the volume of the tubing
    S_bh = 0.0314  # m^2 is the cross section below the injection point
    L_bh = 75  # m is the length below the injection point
    D_bh = 2 * np.sqrt(S_bh / np.pi)
    T_tb = 369.4  # K is the tubing temperature

    D_tb = 0.134  # tubing diameter
    L_tb = 2048  # m, tubing length
    epsilon_tubing = 3e-4

    P_res = 160e5  # 160bar, the constant reservoir pressure

    K_gs = 9.98e-5  # is the gas lift choke constant
    K_inj = 1.40e-4  # is the injection valve choke constant
    K_pr = 2.90e-3  # is the production choke constant
    P_0 = 20e5  # pressure downstream of choke

    # Gas mass fraction from reservoir (paper's constant split)
    alpha_G_m_tb_b = GOR / (GOR + 1)

    # Total tubing control volume (top + bottom)
    V_tot = V_tb + S_bh * L_bh

    # Unpack variables
    m_G_an = y[0]
    m_G_tb = y[1]
    m_L_tb = y[2]
    u1 = u[0]
    u2 = u[1]

    P_tb_b = z[0]  # injection point pressure (tubing at injection)
    P_bh = z[1]  # bottomhole pressure
    w_res = z[2]  # reservoir inflow mass rate (total)

    # -----------------------
    # Annulus equations
    # -----------------------
    P_an_t = R * T_an * m_G_an / (M_G * V_an)
    P_an_b = P_an_t + (m_G_an * g * L_an / V_an)

    rho_G_an_b = P_an_b * M_G / (R * T_an)
    rho_G_in = P_gs * M_G / (R * T_an)

    dP_gs_an = P_gs - P_an_t
    w_G_in = K_gs * u2 * sqrt(rho_G_in * smooth_pos_scaled(dP_gs_an) + eps)

    # -----------------------
    # Tubing gas volume & top pressure (no "liquid-filled bottom")
    # -----------------------
    V_gas_tb_t = V_tot - (m_L_tb / rho_L)
    V_gas_tb_t_safe = smooth_max_scaled(V_gas_tb_t, 1e-6)

    rho_G_tb_t = m_G_tb / V_gas_tb_t_safe
    P_tb_t = rho_G_tb_t * R * T_tb / M_G

    rho_avg_mix_tb = (m_G_tb + m_L_tb) / V_tot
    alpha_avg_L_tb = m_L_tb / (rho_L * V_tot)

    # -----------------------
    # Injection valve (depends on algebraic P_tb)
    # -----------------------
    dP_an_tb = P_an_b - P_tb_b
    w_G_inj = K_inj * sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb) + eps)

    # -----------------------
    # Reservoir split (depends on algebraic w_res)
    # -----------------------
    w_L_res = (1 - alpha_G_m_tb_b) * w_res
    w_G_res = alpha_G_m_tb_b * w_res

    # -----------------------
    # TOP SECTION (inj->top) friction with INSTANTANEOUS w_res and true gas flow
    # -----------------------

    # Equation 11 - Average superficial velocity of liquid phase in tubing
    U_avg_L_tb = 4 * w_L_res / (rho_L * np.pi * D_tb ** 2)
    denom_G = rho_G_tb_t * np.pi * D_tb ** 2
    denom_G_safe = smooth_max_scaled(denom_G, 1e-8)
    U_avg_G_tb = 4 * (w_G_inj + w_G_res) / denom_G_safe
    U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb

    Re_tb = rho_avg_mix_tb * U_avg_mix_tb * D_tb / mu
    Re_tb_safe = smooth_max_scaled(Re_tb, 1.0)  # keep >0 for 6.9/Re

    log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
    log_arg_tb_safe = fmax(log_arg_tb, 1e-12)
    lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2

    F_t = (alpha_avg_L_tb * lambda_tb * rho_avg_mix_tb * U_avg_mix_tb ** 2 * L_tb) / (2 * D_tb)
    dP_t=rho_avg_mix_tb*g*L_tb+F_t

    # -----------------------
    # BOTTOM SECTION (bh->inj) two-phase (not liquid-filled)
    # -----------------------
    rho_G_bh=P_bh*M_G/(R*T_tb)

    denom_h=w_L_res*rho_G_bh+w_G_res*rho_L
    denom_h_safe=smooth_max_scaled(denom_h, 1e-9)
    alpha_L_bh=(w_L_res*rho_G_bh)/denom_h_safe

    rho_mix_bh=alpha_L_bh*rho_L+(1.0-alpha_L_bh)*rho_G_bh

    U_sl_bh=4.0*w_L_res/(rho_L*np.pi*D_bh**2)
    denom_Gbh=rho_G_bh*np.pi*D_bh**2
    denom_Gbh_safe=smooth_max_scaled(denom_Gbh, 1e-9)
    U_sg_bh=4.0*w_G_res/denom_Gbh_safe

    U_m_bh=U_sl_bh+U_sg_bh

    Re_bh = rho_mix_bh*U_m_bh*D_bh/mu
    Re_bh_safe=smooth_max_scaled(Re_bh, 1.0)
    log_arg_bh=(epsilon_tubing/(D_bh*3.7))**1.11+6.9/Re_bh_safe
    log_arg_bh_safe=fmax(log_arg_bh, 1e-9)
    lambda_bh=(1.0/(1.8*log10(log_arg_bh_safe)))**2

    F_bh=(alpha_L_bh*lambda_bh*rho_mix_bh*U_m_bh**2*L_bh)/(2.0*D_bh)
    dP_bh_seg=rho_mix_bh*g*L_bh+F_bh

    # -----------------------
    # Top choke / outlet split (paper-style)
    # -----------------------
    rho_G_tb_b = (P_tb_b * M_G) / (R * T_tb)

    denom_alpha_b = (w_L_res * rho_G_tb_b + (w_G_inj + w_G_res) * rho_L)
    denom_alpha_b_safe = smooth_max_scaled(denom_alpha_b, 1e-9)
    alpha_L_tb_b = (w_L_res * rho_G_tb_b) / denom_alpha_b_safe

    alpha_L_tb_t = 2 * alpha_avg_L_tb - alpha_L_tb_b

    rho_mix_tb_t = alpha_L_tb_t * rho_L + (1 - alpha_L_tb_t) * rho_G_tb_t
    rho_mix_tb_t_safe = smooth_max_scaled(rho_mix_tb_t, 1e-9)

    dP_tb_choke = P_tb_t - P_0
    w_out = K_pr * u1 * sqrt(rho_mix_tb_t_safe * smooth_pos_scaled(dP_tb_choke) + eps)

    denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb_t
    denom_alpha_t_safe = smooth_max_scaled(denom_alpha_t, 1e-9)
    alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb_t) / denom_alpha_t_safe

    w_G_out = alpha_G_tb_t * w_out
    w_L_out = (1.0 - alpha_G_tb_t) * w_out

    Q_out = w_out / rho_mix_tb_t_safe

    # -----------------------
    # DAE algebraic residuals g(x,z,u)=0
    # -----------------------
    # (G1) PI closure: w_res = PI * max(P_res - P_bh, 0)
    dP_res_bh = P_res - P_bh
    g1 = w_res - PI * smooth_pos_scaled(dP_res_bh)

    # (G2) bottomhole pressure closure: P_bh = P_tb + dP_bh_seg
    g2 = P_bh - (P_tb_b + dP_bh_seg)

    # (G3) injection-point pressure closure: P_tb = P_tb_t + dP_t
    g3 = P_tb_b - (P_tb_t + dP_t)

    alg = vertcat(g1, g2, g3)

    # -----------------------
    # Differential equations
    # -----------------------

    dx1 = w_G_in - w_G_inj
    dx2 = w_G_inj + w_G_res - w_G_out
    dx3 = w_L_res - w_L_out
    dx = vertcat(dx1, dx2, dx3)

    # -----------------------
    # Algebraic outputs (ALL)
    # -----------------------
    P_bh_bar = P_bh / 1e5
    P_tb_t_bar = P_tb_t / 1e5
    P_tb_b_bar = P_tb_b / 1e5
    P_an_t_bar = P_an_t / 1e5
    P_an_b_bar = P_an_b / 1e5
    dP_gs_an_bar = dP_gs_an / 1e5
    dP_an_tb_bar = dP_an_tb / 1e5
    dP_res_bh_bar = dP_res_bh / 1e5
    dP_tb_choke_bar = dP_tb_choke / 1e5
    F_t_bar = F_t / 1e5
    F_bh_bar = F_bh / 1e5

    w_w_out = w_L_out * BSW
    w_o_out = w_L_out * (1 - BSW)

    U_avg_L_bh=0
    w_G_in_original=0

    out = vertcat(
        # --- Eq 1-4
        P_an_t_bar,  # 0
        P_an_b_bar,
        rho_G_an_b,
        rho_G_in,

        # --- Eq 5
        dP_gs_an_bar,  # 4
        w_G_in_original,
        w_G_in,

        # --- Eq 6-7
        V_gas_tb_t,  # 7
        V_gas_tb_t_safe,
        rho_G_tb_t,
        P_tb_t_bar,

        # --- Eq 8-13
        rho_avg_mix_tb,  # 11
        alpha_avg_L_tb,
        alpha_G_m_tb_b,
        U_avg_L_tb,
        denom_G,
        denom_G_safe,
        U_avg_G_tb,
        U_avg_mix_tb,

        # --- Eq 14-16
        Re_tb,  # 19
        Re_tb_safe,
        log_arg_tb,
        log_arg_tb_safe,
        lambda_tb,
        F_t_bar,

        # --- Eq 17-18
        P_tb_b_bar,  # 25
        dP_an_tb_bar,
        w_G_inj,

        # --- Eq 19-23
        U_avg_L_bh,  # 28
        Re_bh,
        log_arg_bh,
        lambda_bh,
        F_bh_bar,
        P_bh_bar,

        # --- Eq 24-27
        dP_res_bh_bar,  # 34
        w_res,
        w_L_res,
        w_G_res,
        rho_G_tb_b,

        # --- Eq 28-30
        denom_alpha_b,  # 39
        denom_alpha_b_safe,
        alpha_L_tb_b,
        alpha_L_tb_t,
        rho_mix_tb_t,
        rho_mix_tb_t_safe,

        # --- Eq 31-35
        dP_tb_choke_bar,  # 45
        w_out,
        Q_out,
        denom_alpha_t,
        denom_alpha_t_safe,
        alpha_G_tb_t,
        w_G_out,
        w_L_out,

        w_w_out,
        w_o_out,

        rho_mix_bh

    )

    return dx, alg, out


Z_NAMES = [
    # --- Eq 1-4
    "P_an_t_bar",
    "P_an_b_bar",
    "rho_G_an_b",
    "rho_G_in",

    # --- Eq 5
    "dP_gs_an_bar",
    "w_G_in_original",
    "w_G_in",

    # --- Eq 6-7
    "V_gas_tb_t",
    "V_gas_tb_t_safe",
    "rho_G_tb_t",
    "P_tb_t_bar",

    # --- Eq 8-13
    "rho_avg_mix_tb",
    "alpha_avg_L_tb",
    "alpha_G_m_tb_b",
    "U_avg_L_tb",
    "denom_G",
    "denom_G_safe",
    "U_avg_G_tb",
    "U_avg_mix_tb",

    # --- Eq 14-16
    "Re_tb",
    "Re_tb_safe",
    "log_arg_tb",
    "log_arg_tb_safe",
    "lambda_tb",
    "F_t_bar",

    # --- Eq 17-18
    "P_tb_b_bar",
    "dP_an_tb_bar",
    "w_G_inj",

    # --- Eq 19-23
    "U_avg_L_bh",
    "Re_bh",
    "log_arg_bh",
    "lambda_bh",
    "F_bh_bar",
    "P_bh_bar",

    # --- Eq 24-27
    "dP_res_bh_bar",
    "w_res",
    "w_L_res",
    "w_G_res",
    "rho_G_tb_b",

    # --- Eq 28-30
    "denom_alpha_b",
    "denom_alpha_b_safe",
    "alpha_L_tb_b",
    "alpha_L_tb_t",
    "rho_mix_tb_t",
    "rho_mix_tb_t_safe",

    # --- Eq 31-35
    "dP_tb_choke_bar",
    "w_out",
    "Q_out",
    "denom_alpha_t",
    "denom_alpha_t_safe",
    "alpha_G_tb_t",
    "w_G_out",
    "w_L_out",

    "w_w_out",
    "w_o_out",

    "rho_mix_bh"
]

def build_well_model(i: int, name_prefix="well"):
    """
    Returns a dict with explicit metadata + a compiled CasADi function:
        F_all(y,z,u) -> (dx, g, out)
    """
    well_id = i + 1
    fname = f"glc_well_{well_id:02d}_rigorous_casadi"
    if fname not in globals():
        raise AttributeError(f"No function '{fname}' in this module.")
    well_func = globals()[fname]

    nx = 3
    nu = 2
    nz = 3   # your algebraic vector length, e.g. [P_tb_b, P_bh, w_res]

    y = ca.MX.sym(f"y_{name_prefix}_{well_id}", nx)
    z = ca.MX.sym(f"z_{name_prefix}_{well_id}", nz)
    u = ca.MX.sym(f"u_{name_prefix}_{well_id}", nu)

    dx, g, out = well_func(y, z, u)

    F_all = ca.Function(
        f"F_all_{name_prefix}_{well_id}",
        [y, z, u],
        [dx, g, out],
        ["y", "z", "u"],
        ["dx", "g", "out"]
    )

    return {
        "is_dae": True,
        "nx": nx,
        "nu": nu,
        "nz": nz,
        "Z_NAMES": Z_NAMES,
        "F_all": F_all,
    }