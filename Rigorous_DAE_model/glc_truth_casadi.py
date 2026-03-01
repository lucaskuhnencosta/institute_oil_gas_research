from casadi import *
import numpy as np
import casadi as ca

def make_glc_well_rigorous(BSW,GOR,PI):

    def glc_well(y, z, u):

        def softplus_stable(x):
            ax = sqrt(x * x + eps)
            return 0.5 * (x + ax) + log(1 + exp(-ax))

        def smooth_pos_scaled(dp_pa):
            x = (k_pos * dp_pa)
            return (1 / k_pos) * (softplus_stable(x))

        def smooth_max_scaled(z, zmin):
            # smooth approximation of max(z,zmin))
            return zmin + smooth_pos_scaled(z - zmin)

        k_pos = 20
        eps = 1e-12

        # # Well properties ###
        # BSW = 0
        # GOR = 0 # is the gas oil ratio
        # PI = 3.00e-6  # is the productivity index in kg/(s.Pa)

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
        K_inj = 1.40e-4  # is the injection valve choke constant
        K_pr = 2.90e-3  # is the production choke constant
        K0_int=0.20

        # Friction
        epsilon_tubing = 3e-4

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

        # Algebraic variables

        P_bh_g=z[0] #Pressure bottomhole, closed at g6
        P_tb_b_g=z[1] # Pressure at the bottom of the tubing, closed at g7
        w_res_g=z[2] # Reservoir flow in kg/s
        w_up_g=z[3] # Flow into the tubing in kg/s


        # -----------------------
        # PART 1 - RESERVOIR INFLOW
        # -----------------------

        w_L_res=w_res_g/(1+GOR) # 1.1
        w_o_res=(1-BSW)*w_L_res # 1.2
        w_w_res=BSW*w_L_res  #1.3
        w_G_res=GOR*w_L_res #1.4

        # -----------------------
        # PART 2 - ANNULUS (from state)
        # -----------------------
        P_an_t=R*T_an*m_G_an/(M_G*V_an) # 2.1
        P_an_b=P_an_t+(m_G_an*g*L_an/V_an) # 2.2

        rho_G_an_b=P_an_b*M_G/(R*T_an) # 2.3
        rho_G_in=P_gs*M_G/(R*T_an) # 2.4

        dP_gs_an = P_gs - P_an_t # 2.5
        w_G_in = K_gs * u2 * sqrt(rho_G_in * smooth_pos_scaled(dP_gs_an) + eps) # 2.6

        # This makes our equation 2.7
        dP_an_tb = P_an_b - P_tb_b_g
        w_G_inj = K_inj * sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb) + eps)

        # -----------------------
        # PART 3 - DENSITIES USING STATES
        # -----------------------

        V_G_tb=V_tb-(m_o_t/rho_o)-(m_w_t/rho_w)
        V_G_tb_safe = smooth_max_scaled(V_G_tb, 1e-6)

        rho_G_tb = m_G_t/V_G_tb_safe

        V_G_bh=V_bh-(m_o_b/rho_o)-(m_w_b/rho_w)
        V_G_bh_safe=smooth_max_scaled(V_G_bh, 1e-6)

        rho_G_bh=m_G_b/V_G_bh_safe

        P_tb_t = rho_G_tb * R * T_tb / M_G
        P_bh_t= rho_G_bh * R * T_bh / M_G

        # -----------------------
        # PART 4 - Hold-ups from states
        # -----------------------

        V_w_tb_states = m_w_t / rho_w
        V_o_tb_states = m_o_t / rho_o
        V_L_tb_states=V_w_tb_states+V_o_tb_states

        V_w_bh_states = m_w_b / rho_w
        V_o_bh_states = m_o_b / rho_o
        V_L_bh_states=V_w_bh_states+V_o_bh_states

        alpha_L_tb=(V_w_tb_states+V_o_tb_states)/V_tb
        alpha_G_tb=1.0-alpha_L_tb

        alpha_L_bh=(V_w_bh_states+V_o_bh_states)/V_bh
        alpha_G_bh=1.0-alpha_L_bh

        V_L_bh_states_safe = smooth_max_scaled(V_L_bh_states, 1e-9)
        V_L_tb_states_safe = smooth_max_scaled(V_L_tb_states, 1e-9)

        # Liquid compositions in each CV (for splitting oil/water)
        f_w_b = V_w_bh_states / V_L_bh_states_safe
        f_w_t = V_w_tb_states / V_L_tb_states_safe

        # Densities in each CV (use constant rho_L in spirit of article; you can swap to state-based if desired)
        # Using constant rho_L keeps consistency with the article’s closure style.
        rho_L_tb = rho_L
        rho_L_bh = rho_L

        # Mixture density in each CV (volume averaged)
        rho_mix_tb = alpha_L_tb * rho_L_tb + alpha_G_tb * rho_G_tb
        rho_mix_bh = alpha_L_bh * rho_L_bh + alpha_G_bh * rho_G_bh
        rho_mix_tb_safe = smooth_max_scaled(rho_mix_tb, 1e-9)
        rho_mix_bh_safe = smooth_max_scaled(rho_mix_bh, 1e-9)

        # -----------------------
        # PART 5 - Decomposition of w_up
        # -----------------------

        alpha_G_bh_mass = (alpha_G_bh * rho_G_bh) / rho_mix_bh_safe
        alpha_G_bh_mass = ca.fmin(ca.fmax(alpha_G_bh_mass, 0.0), 1.0)  # keep bounded

        w_G_up = alpha_G_bh_mass * w_up_g
        w_L_up = (1.0 - alpha_G_bh_mass) * w_up_g

        w_w_up = f_w_b * w_L_up
        w_o_up = (1.0 - f_w_b) * w_L_up

        # -----------------------
        # PART 6 - Superficial velocities
        # -----------------------

        # Superficial velocities bottom
        U_sg_b=w_G_res/(rho_G_bh*S_bh) # 4.1
        U_sl_b=w_L_up/(rho_L_bh*S_bh) # 4.2

        # Superficial velocities top
        U_sg_t=(w_G_up+w_G_inj)/(rho_G_tb*S_tb) # 4.3
        U_sl_t=(w_L_up)/(rho_L_tb*S_tb) # 4.4

        U_avg_b=U_sg_b+U_sl_b # 4.5
        U_avg_t=U_sg_t+U_sl_t # 4.6

        # -----------------------
        # PART 7 - Friction and gravity
        # -----------------------

        # In the top section
        Re_tb = rho_mix_tb_safe*U_avg_t*D_tb/mu
        Re_tb_safe=smooth_max_scaled(Re_tb,1.0)  # keep >0 for 6.9/Re

        log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
        log_arg_tb_safe = fmax(log_arg_tb, 1e-12)
        lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2

        F_t = (lambda_tb*rho_mix_tb*U_avg_t**2*L_tb)/(2*D_tb)

        dP_t=rho_mix_tb*g*L_tb+F_t
        P_tb_b=P_tb_t+dP_t

        g1=P_tb_b-P_tb_b_g # Here we close the loop for the tubing bottom pressure

        # In the bottom section
        Re_bh = rho_mix_bh_safe*U_avg_b*D_bh/mu
        Re_bh_safe=smooth_max_scaled(Re_bh, 1.0)

        log_arg_bh=(epsilon_tubing/(D_bh*3.7))**1.11+6.9/Re_bh_safe
        log_arg_bh_safe=fmax(log_arg_bh, 1e-12)
        lambda_bh=(1.0/(1.8*log10(log_arg_bh_safe)))**2

        F_bh=(lambda_bh*rho_mix_bh*U_avg_b**2*L_bh)/(2.0*D_bh)

        dP_bh_seg=rho_mix_bh*g*L_bh+F_bh
        P_bh=P_tb_b+dP_bh_seg

        g2= P_bh-P_bh_g

        # -----------------------
        # PART 8 - Interface minor loss
        # -----------------------
        A_b=S_bh
        A_t=S_tb #smaller
        beta=A_t/A_b #area ratio

        # Simple diameter-based minor-loss coefficient (smooth-ish & safe):
        # You can tune K0_int to make ΔP small.
        # - For contraction: (1/beta - 1)^2
        K_con = (1.0/beta - 1.0)**2
        K_int = K0_int + K_con

        # INTERFACE HERE!
        dP_int = K_int*(w_up_g**2)/(2.0*rho_mix_bh*(A_t**2)+eps)

        g3=(P_bh_t-P_tb_b)-dP_int

        # -----------------------
        # PART 9 — Outflow at top choke (article-style)
        # -----------------------

        rho_G_tb_b=P_tb_b*M_G/(R*T_tb)

        denom_alpha_b=(w_L_up*rho_G_tb_b+(w_G_inj+w_G_up)*rho_L_tb)
        denom_alpha_b_safe=smooth_max_scaled(denom_alpha_b, 1e-9)

        alpha_L_tb_b=w_L_up*rho_G_tb_b/denom_alpha_b_safe
        alpha_L_tb_t = 2 * alpha_L_tb - alpha_L_tb_b

        # Equation 30 - Mixture density at the top
        rho_mix_tb_t = alpha_L_tb_t * rho_L + (1 - alpha_L_tb_t) * rho_G_tb
        rho_mix_tb_t_safe = smooth_max_scaled(rho_mix_tb_t, 1e-9)

        dP_tb_choke = P_tb_t - P_0
        rho_G_tb_b=P_tb_b*M_G/(R*T_tb)
        w_out = K_pr * u1 * ca.sqrt(rho_mix_tb_t_safe * smooth_pos_scaled(dP_tb_choke) + eps)

        denom_alpha_t = alpha_L_tb_t * rho_L + (1.0 - alpha_L_tb_t) * rho_G_tb
        denom_alpha_t_safe = fmax(denom_alpha_t, 1e-12)

        # Equation 33 - Gas Mass Fraction at top of tubing
        alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb) / denom_alpha_t_safe

        w_G_out = alpha_G_tb_t * w_out

        w_L_out = (1.0 - alpha_G_tb_t) * w_out

        w_w_out = f_w_t * w_L_out
        w_o_out = (1.0 - f_w_t) * w_L_out

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
        w_res=PI*smooth_pos_scaled(dP_res_bh) # This is closure for w_res
        g4=w_res_g-w_res


        alg = vertcat(g1, g2,g3,g4)

        P_bh_bar = P_bh / 1e5
        P_tb_t_bar = P_tb_t / 1e5
        P_tb_b_bar = P_tb_b / 1e5

        P_hidro_tb_bar=(rho_mix_tb*g*L_tb)/1e5
        P_hidro_bh_bar=(rho_mix_bh*g*L_bh)/1e5

        dP_int_bar=dP_int / 1e5
        dP_gs_an_bar = dP_gs_an / 1e5
        dP_an_tb_bar = dP_an_tb / 1e5
        dP_res_bh_bar = dP_res_bh / 1e5
        dP_tb_choke_bar = dP_tb_choke / 1e5
        F_t_bar = F_t / 1e5
        F_bh_bar = F_bh / 1e5

        out = ca.vertcat(

            # =================================
            # States (masses in kg)
            # =================================
            m_G_an,
            m_G_t,
            m_o_t,
            m_w_t,
            m_G_b,
            m_o_b,
            m_w_b,

            # =================================
            # Hold-ups / Phase volumes (m³)
            # =================================
            V_L_tb_states,

            V_L_bh_states,

            alpha_L_tb,
            alpha_G_tb,
            alpha_L_tb_t,
            alpha_L_tb_b,

            alpha_L_bh,
            alpha_G_bh,

            # =================================
            # Pressures (bar)
            # =================================
            P_bh_bar,
            P_tb_t_bar,
            P_tb_b_bar,
            P_hidro_tb_bar,
            P_hidro_bh_bar,

            # Friction (bar)
            F_t_bar,
            F_bh_bar,

            # =================================
            # Delta Pressures (bar)
            # =================================
            dP_int_bar,
            dP_gs_an_bar,
            dP_an_tb_bar,
            dP_res_bh_bar,
            dP_tb_choke_bar,

            # =================================
            # Flows (kg/s)
            # =================================
            w_res,
            w_L_res,
            w_G_res,

            w_out,
            w_L_out,
            w_w_out,
            w_o_out,

            w_G_inj,
            w_up_g
        )
        return dx, alg, out

    return glc_well

Z_NAMES = [

    # =================================
    # States (kg)
    # =================================
    "m_G_an",
    "m_G_t",
    "m_o_t",
    "m_w_t",
    "m_G_b",
    "m_o_b",
    "m_w_b",

    # =================================
    # Hold-ups / Volumes (m³)
    # =================================
    "V_L_tb_states",

    "V_L_bh_states",
    # Volume fractions
    "alpha_L_tb",
    "alpha_G_tb",
    "alpha_L_tb_t",
    "alpha_L_tb_b",
    "alpha_L_bh",
    "alpha_G_bh",


    # =================================
    # Pressures (bar)
    # =================================
    "P_bh_bar",
    "P_tb_t_bar",
    "P_tb_b_bar",
    "P_hidro_tb_bar",
    "P_hidro_bh_bar",

    # Friction (bar)
    "F_t_bar",
    "F_bh_bar",

    # =================================
    # Delta Pressures (bar)
    # =================================
    "dP_int_bar",
    "dP_gs_an_bar",
    "dP_an_tb_bar",
    "dP_res_bh_bar",
    "dP_tb_choke_bar",

    # =================================
    # Flows (kg/s)
    # =================================
    "w_res",
    "w_L_res",
    "w_G_res",

    "w_out",
    "w_L_out",
    "w_w_out",
    "w_o_out",

    "w_G_inj",

    "w_up"
]



#
# def build_well_model(i: int, name_prefix="well"):
#     """
#     Returns a dict with explicit metadata + a compiled CasADi function:
#         F_all(y,z,u) -> (dx, g, out)
#     """
#     well_id = i + 1
#     fname = f"glc_well_{well_id:02d}_rigorous_casadi"
#     if fname not in globals():
#         raise AttributeError(f"No function '{fname}' in this module.")
#     well_func = globals()[fname]
#
#     nx = 7
#     nu = 2
#     nz = 8   # your algebraic vector length, e.g. [P_tb_b, P_bh, w_res]
#
#     y = ca.MX.sym(f"y_{name_prefix}_{well_id}", nx)
#     z = ca.MX.sym(f"z_{name_prefix}_{well_id}", nz)
#     u = ca.MX.sym(f"u_{name_prefix}_{well_id}", nu)
#
#     dx, g, out = well_func(y, z, u)
#
#     F_all = ca.Function(
#         f"F_all_{name_prefix}_{well_id}",
#         [y, z, u],
#         [dx, g, out],
#         ["y", "z", "u"],
#         ["dx", "g", "out"]
#     )
#
#     return {
#         "is_dae": True,
#         "nx": nx,
#         "nu": nu,
#         "nz": nz,
#         "Z_NAMES": Z_NAMES,
#         "F_all": F_all,
#     }