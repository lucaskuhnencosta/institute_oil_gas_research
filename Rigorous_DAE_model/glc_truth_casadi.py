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

    # Well properties
    BSW = 0
    GOR = 0 # is the gas oil ratio
    PI = 3.00e-6  # is the productivity index in kg/(s.Pa)

    # Constants (general)
    R = 8.314  # J/(K*mol) is the universal gas constant
    g = 9.81  # m/s^2 is the gravity
    mu_o = 3.64e-3  # Pa.s is the viscosity

    # Constants (fluid)
    mu_w = 1.00e-3
    rho_o = 760  # kg/m^3 is the density of the liquid in the tubing
    rho_w = 1000
    rho_L = 1.0 / (BSW / rho_w + (1.0 - BSW) / rho_o)
    mu = np.exp((1 - BSW) * np.log(mu_o) + BSW * np.log(mu_w))
    M_G = 0.0167  # (kg/mol) is the gas molecular weight

    # Temperatures
    T_an = 348  # K is the annulus temperature
    T_tb = 369.4  # K is the tubing temperature
    T_b=371.5


    # Volumes, lengths and areas
    ### Annulus ###
    V_an = 64.34  # m^3 is the annulus volume
    L_an = 2048  # m is the length of the annulus

    ### Tubing bottom
    L_bh = 75  # m is the length below the injection point
    S_bh = 0.0314  # m^2 is the cross section below the injection point
    D_bh = 2 * np.sqrt(S_bh / np.pi) # 0.2 m diameter
    V_b = S_bh * L_bh

    ### Tubing top
    L_tb_t = 1973
    D_tb = 0.134  # 0.13m diameter
    S_tb_t=(np.pi*D_tb**2)/4
    V_tb_t=S_tb_t*L_tb_t


    # Pressures
    P_gs = 140e5  # 140bar is the gas source pressure
    P_res = 160e5  # 160bar, the constant reservoir pressure
    P_0 = 20e5  # pressure downstream of choke

    # Chokes
    K_gs = 9.98e-5  # is the gas lift choke constant
    K_inj = 1.40e-4  # is the injection valve choke constant
    K_pr = 2.90e-3  # is the production choke constant
    K0_int=0.2

    # Friction
    epsilon_tubing = 3e-4

    # Slip model
    C_0_b=1.15
    C_0_t=1.20

    V_d_b=0.25
    V_d_t=0.40


    # ---------- unpack ----------
    m_G_an=y[0]
    m_G_t=y[1]
    m_o_t=y[2]
    m_w_t=y[3]
    m_G_b=y[4]
    m_o_b=y[5]
    m_w_b =y[6]

    u1=u[0]
    u2=u[1]

    P_bh=z[0]
    P_tb_b=z[1]
    w_res=z[2]
    w_up=z[3]

    w_up_pos=smooth_pos_scaled(w_up)

    # -----------------------
    # PART 1 - RESERVOIR INFLOW
    # -----------------------

    w_l_res=w_res/(1+GOR) # 1.1
    w_o_res=(1-BSW)*w_l_res # 1.2
    w_w_res=BSW*w_l_res  #1.3
    w_G_res=GOR*w_l_res #1.4

    # -----------------------
    # PART 2 - ANNULUS
    # -----------------------
    P_an_t=R*T_an*m_G_an/(M_G*V_an) # 2.1
    P_an_b=P_an_t+(m_G_an*g*L_an/V_an) # 2.2

    rho_G_an_b=P_an_b*M_G/(R*T_an) # 2.3
    rho_G_in=P_gs*M_G/(R*T_an) # 2.4

    dP_gs_an = P_gs - P_an_t # 2.5
    w_G_in = K_gs * u2 * sqrt(rho_G_in * smooth_pos_scaled(dP_gs_an) + eps) # 2.6

    # We end this cycle by using the "guessed" p_tb_b to find a w_G_inj
    # This makes our equation 2.7
    dP_an_tb = P_an_b - P_tb_b
    w_G_inj = K_inj * sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb) + eps)

    # -----------------------
    # PART 3 - DENSITIES OF GAS AND PRESSURES FROM DENSITIES
    # -----------------------

    # Equation 3.1
    V_G_t=V_tb_t-(m_o_t/rho_o)-(m_w_t/rho_w)
    V_G_t_safe=smooth_max_scaled(V_G_t,1e-6)
    rho_G_tb=m_G_t/V_G_t_safe

    # Equation 3.2
    V_G_b=V_b-(m_o_b/rho_o)-(m_w_b/rho_w)
    V_G_b_safe=smooth_max_scaled(V_G_b,1e-6)
    rho_G_b=m_G_b/V_G_b_safe

    P_tb_t=rho_G_tb*R*T_tb/M_G # 3.3 top of the top tubing section
    P_b_t=rho_G_b*R*T_b/M_G # 3.4 top of the bottom section calculating P_b_t explicitly

    # Now we need to compute state implied hold-ups
    H_g_t_state=V_G_t_safe/V_tb_t
    H_g_b_state=V_G_b_safe/V_b

    # -----------------------
    # PART 4 - Drift
    # -----------------------
    # Bottom composition from states (volumetric water fraction inside bottom liquid)
    V_w_b=m_w_b/rho_w
    V_o_b=m_o_b/rho_o
    f_w_b_state=V_w_b/smooth_max_scaled(V_w_b+V_o_b,1e-9)

    # Convert bottom holdup to gas MASS fraction
    # Superficial velocities bottom
    denom_massfrac_b=H_g_b_state*rho_G_b+(1.0-H_g_b_state)*rho_L
    xG_b=(H_g_b_state*rho_G_b)/smooth_max_scaled(denom_massfrac_b,1e-9)
    xL_b=1.0-xG_b

    # Split interface mass flow into phases (computed, not guesses)
    w_G_up=xG_b*w_up_pos
    w_L_up=xL_b*w_up_pos
    w_w_up=f_w_b_state*w_L_up
    w_o_up=(1.0-f_w_b_state)*w_L_up

    # Superficial velocities bottom
    U_sg_b=w_G_res/(rho_G_b*S_bh) # 4.1
    U_sl_b=w_l_res/(rho_L*S_bh) # 4.2

    # Superficial velocities top
    U_sg_t=(w_G_up+w_G_inj)/(rho_G_tb*S_tb_t) # 4.3
    U_sl_t=(w_L_up)/(rho_L*S_tb_t) # 4.4

    U_avg_b=U_sg_b+U_sl_b # 4.5
    U_avg_t=U_sg_t+U_sl_t # 4.6

    U_g_b=C_0_b*U_avg_b+V_d_b # 4.7
    U_g_t=C_0_t*U_avg_t+V_d_t # 4.8

    H_g_b=U_sg_b/smooth_max_scaled(U_g_b,1e-6) # 4.9
    H_g_t=U_sg_t/smooth_max_scaled(U_g_t,1e-6) # 4.10

    H_l_b=1.0-H_g_b # 4.11
    H_l_t=1.0-H_g_t # 4.12

    f_w_b = (w_w_res / rho_w) / ((w_w_res / rho_w) + (w_o_res / rho_o) + eps)
    H_w_b=H_l_b*f_w_b # 4.13
    H_o_b=H_l_b-H_w_b # 4.14

    f_w_t = (w_w_up / rho_w) / ((w_w_up / rho_w) + (w_o_up / rho_o) + eps)
    H_w_t=H_l_t*f_w_t# 4.15
    H_o_t=H_l_t-H_w_t # 4.16

    # Mixture densities and their safety version
    rho_mix_tb=H_g_t*rho_G_tb+H_w_t*rho_w+H_o_t*rho_o # 4.17
    rho_mix_bh=H_g_b*rho_G_b+H_w_b*rho_w+H_o_b*rho_o # 4.18

    rho_mix_tb_safe=smooth_max_scaled(rho_mix_tb,1e-6)
    rho_mix_bh_safe=smooth_max_scaled(rho_mix_bh,1e-6)

    # -----------------------
    # PART 5 - Friction
    # -----------------------

    # In the top section

    Re_tb = rho_mix_tb_safe*U_avg_t*D_tb/mu
    Re_tb_safe=smooth_max_scaled(Re_tb,1.0)  # keep >0 for 6.9/Re

    log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
    log_arg_tb_safe = fmax(log_arg_tb, 1e-12)
    lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2

    F_t = (lambda_tb*rho_mix_tb_safe*U_avg_t**2*L_tb_t)/(2*D_tb)

    dP_t=rho_mix_tb_safe*g*L_tb_t+F_t

    # In the bottom section

    Re_bh = rho_mix_bh_safe*U_avg_b*D_bh/mu
    Re_bh_safe=smooth_max_scaled(Re_bh, 1.0)

    log_arg_bh=(epsilon_tubing/(D_bh*3.7))**1.11+6.9/Re_bh_safe
    log_arg_bh_safe=fmax(log_arg_bh, 1e-12)
    lambda_bh=(1.0/(1.8*log10(log_arg_bh_safe)))**2

    F_bh=(lambda_bh*rho_mix_bh_safe*U_avg_b**2*L_bh)/(2.0*D_bh)

    dP_bh_seg=rho_mix_bh_safe*g*L_bh+F_bh

    # -----------------------
    # PART 6 - Interface minor loss
    # -----------------------
    A_b=S_bh
    A_t=S_tb_t
    beta=A_t/A_b #area ratio


    # Simple diameter-based minor-loss coefficient (smooth-ish & safe):
    # You can tune K0_int to make ΔP small.
    # - For expansion: (1 - beta)^2
    # - For contraction: (1/beta - 1)^2
    K_exp = (1.0 - beta)**2
    K_con = (1.0/beta - 1.0)**2
    K_int = K0_int + ca.if_else(beta >= 1.0, K_exp, K_con)

    # ΔP_int in mass-flow form:
    # ΔP = K * w_up^2 / (2 rho A^2)
    dP_int = K_int * (w_up_pos**2) / (2.0 * rho_mix_bh_safe * (A_t**2) + eps)

    # -----------------------
    # PART 7 - Outflow and outflow split
    # -----------------------

    dP_tb_choke=P_tb_t-P_0 # 6.1
    w_out=K_pr*u1*sqrt(rho_mix_tb_safe*smooth_pos_scaled(dP_tb_choke)+eps) # 6.2

    Q_out=w_out/rho_mix_tb_safe
    Q_G_out=H_g_t*Q_out
    Q_L_out=(1.0-H_g_t)*Q_out

    rho_L_out=1.0/(f_w_t/rho_w+(1.0-f_w_t)/rho_o)

    w_G_out=rho_G_tb*Q_G_out
    w_L_out=rho_L_out*Q_L_out
    w_w_out=f_w_t*w_L_out
    w_o_out=(1.0-f_w_t)*w_L_out

    # ============================================================
    # PART 8 — ODEs (states evolve)
    # ============================================================
    # Annulus gas
    dx_mG_an = w_G_in - w_G_inj

    # Bottom CV
    dx_mG_b  = w_G_res - w_G_up
    dx_mo_b  = w_o_res - w_o_up
    dx_mw_b  = w_w_res - w_w_up

    # Top CV (injection enters top)
    dx_mG_t  = w_G_up + w_G_inj - w_G_out
    dx_mo_t  = w_o_up           - w_o_out
    dx_mw_t  = w_w_up           - w_w_out

    dx = ca.vertcat(dx_mG_an, dx_mG_t, dx_mo_t, dx_mw_t, dx_mG_b, dx_mo_b, dx_mw_b)

    # ============================================================
    # PART 9 — Algebraic residuals g(x,z,u)=0
    # ============================================================

    # (G1) PI closure: w_res = PI * max(P_res - P_bh, 0)
    dP_res_bh = P_res - P_bh
    g1 = w_res - PI * smooth_pos_scaled(dP_res_bh)

    # (G2) bottomhole pressure closure: P_bh = P_tb + dP_bh_seg
    g2 = P_bh - (P_tb_b + dP_bh_seg)

    # (G3) injection-point pressure closure: P_tb = P_tb_t + dP_t
    g3 = P_tb_b - (P_tb_t + dP_t)

    # (G4) connecting tubing parts through pressure
    g4=(P_tb_b-P_b_t)-dP_int

    H_g_t_drift=H_g_t
    H_g_b_drift=H_g_b

    # (G5-6) holdup consistency constraints (state holdup == drift holdup)
    g5 = H_g_t_state - H_g_t_drift
    g6 = H_g_b_state - H_g_b_drift

    alg = vertcat(g1, g2, g3,g4,g5,g6)

    # Differential equations
    # -----------------------


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