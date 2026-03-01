from casadi import *
import numpy as np
import casadi as ca

def make_glc_well_surrogate(BSW,GOR,PI):

    def glc_well(y,u):
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
            # softplus(x) = max(x,0) + log(1+exp(-|x|)), stable for large x
            ax = sqrt(x * x + eps)
            return 0.5 * (x + ax) + log(1 + exp(-ax))

        def smooth_pos_scaled(dp_pa,scale=1):
            x=(k_pos*dp_pa)/scale
            # softplus(z) = (1/k) log(1+exp(k z))
            return (scale/k_pos)*(softplus_stable(x))

        def smooth_max_scaled(z,zmin,scale=1):
            # smooth approximation of max(z,zmin))
            return zmin + smooth_pos_scaled(z - zmin,scale)

        k_pos=20
        eps=1e-12

        # Well properties ###
        w_avg_res = 18 #kg/s is an average flow from reservoir to compute the friction terms and model bottomhole pressure to calculate the actual flow from reservoir

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
        K0_int=0.01

        # Friction
        epsilon_tubing = 3e-4

        # ---------- unpack ----------
        m_G_an = y[0]
        m_G_tb = y[1]
        m_L_tb = y[2]
        u1 = u[0]
        u2 = u[1]

        # -----------------------
        # PART 1 - RESERVOIR INFLOW
        # -----------------------

        # -----------------------
        # PART 2 - ANNULUS (from state)
        # -----------------------

        P_an_t=R*T_an*m_G_an/(M_G*V_an)
        P_an_b=P_an_t+(m_G_an*g*L_an/V_an)

        rho_G_an_b=P_an_b*M_G/(R*T_an)
        rho_G_in=P_gs*M_G/(R*T_an)

        # Equation 5 - Mass flow into annulus using Bernoulli's equation for orifice flow
        dP_gs_an=P_gs-P_an_t
        w_G_in=K_gs*u2*sqrt(rho_G_in*smooth_pos_scaled(dP_gs_an,scale=1)+eps)

        ###
        ###

        # -----------------------
        # PART 3 - DENSITIES USING STATES
        # -----------------------

        V_gas_tb_t = (V_tb + V_bh) - (m_L_tb / rho_L)
        V_gas_tb_t_safe = smooth_max_scaled(V_gas_tb_t, 1e-6)

        rho_G_tb_t=m_G_tb/V_gas_tb_t_safe

        ###
        ###

        ###

        P_tb_t = rho_G_tb_t * R * T_tb / M_G
        ###

        # -----------------------
        # PART 4 - Hold-ups from states
        # -----------------------

        alpha_avg_L_tb = (m_L_tb - rho_L * S_bh * L_bh) / (V_tb * rho_L)


        # Equation 8 - Average mixture density inside tubing (above injection point)
        rho_avg_mix_tb=(m_G_tb+m_L_tb-rho_L*S_bh*L_bh)/V_tb

        # Equation 9 - Average liquid volume fraction inside tubing (above injection point)


        # Equation 10 - Gas mass fraction at bottom-hole, as the GOR is constant this is also constant and not an algebraic equation
        alpha_G_tb_b = GOR / (GOR + 1)

        ###

        # Equation 11 - Average superficial velocity of liquid phase in tubing
        U_avg_L_tb=(4*(1-alpha_G_tb_b)*w_avg_res/(rho_L*np.pi*D_tb**2))


        denom_G=rho_G_tb_t * np.pi * D_tb ** 2
        denom_G_safe=smooth_max_scaled(denom_G,1e-8)
        # Equation 12 - Average superficial velocity of gas phase
        U_avg_G_tb=4*(w_G_in+alpha_G_tb_b*w_avg_res)/denom_G_safe

        # Equation 13 - Average mixture velocity in tubing
        U_avg_mix_tb = U_avg_L_tb + U_avg_G_tb

        # Equation 14 - Reynolds number of flow in tubing
        Re_tb=rho_avg_mix_tb*U_avg_mix_tb*D_tb/mu
        Re_tb_safe = smooth_max_scaled(Re_tb, 1.0)  # keep >0 for 6.9/Re

        # Equation 15 - Friction factor in tubing

        log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
        log_arg_tb_safe = fmax(log_arg_tb, 1e-12)

        # log_arg_tb_safe = smooth_max_scaled(log_arg_tb, 1e-12)


        lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2

        # # THIS EQUATION IS DIFFERENT DEPENDING IF YOU ADOPT THIS VERSION OF THE COMPLETE VERSION!
        # Re_tb = fmax(Re_tb, 13000.0)
        # Re_tb = fmin(Re_tb, 115000.0)
        # lambda_tb = -1.78223894e-17 * Re_tb ** 3 + 4.56100539e-12 * Re_tb ** 2 - 4.18248919e-07 * Re_tb + 3.29432465e-02

        # Equation 16 - Pressure loss due to friction in tubing
        F_t=(alpha_avg_L_tb*lambda_tb*rho_avg_mix_tb*U_avg_mix_tb**2*L_tb)/(2*D_tb)

        # Equation 17 - Pressure at tubing bottom
        P_tb_b=P_tb_t+rho_avg_mix_tb*g*L_tb+F_t


        dP_an_tb=P_an_b-P_tb_b

        # Equation 18 - Mass flow rate of gas injected into tubing
        w_G_inj=K_inj*sqrt(rho_G_an_b * smooth_pos_scaled(dP_an_tb,scale=1)+eps)

        # Equation 19 - Liquid velocity at bottom-hole
        U_avg_L_bh=w_avg_res/(rho_L*S_bh)

        # Equation 20 - Reynolds number of flow at bottom-hole
        Re_bh=rho_L*U_avg_L_bh*D_bh/mu

        # log_arg_bh=(epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh
        # # log_arg_bh_safe=max(float(log_arg_bh), 1e-12) if not hasattr(log_arg_bh,"is_symbolic") else log_arg_bh
        # # if hasattr(log_arg_bh,"is_symbolic"):
        # log_arg_bh_safe=smooth_max(log_arg_bh, 1e-12)

        # Equation 21 - Friction factor at bottom hole
        log_arg_bh = (epsilon_tubing / (D_bh * 3.7)) ** 1.11 + 6.9 / Re_bh
        lambda_bh = (1 / (1.8 * log10(log_arg_bh))) ** 2

        # lambda_bh = (1 / (1.8 * log10(log_arg_bh_safe))) ** 2

        # Equation 22 - Pressure loss due to friction from bottom-hole to injection point
        F_bh=(lambda_bh*rho_L*U_avg_L_bh**2*L_bh)/(2.0*D_bh)

        # Equation 23 - Pressure at bottom-hole
        P_bh=P_tb_b+F_bh+rho_L*g*L_bh

        # Equation 24 - Mass flow rate from reservoir to tubing
        dP_res_bh=P_res-P_bh
        w_res = PI * smooth_pos_scaled(dP_res_bh,scale=1)

        # Equation 25 - Mass flow rate of liquid from reservoir to tubing
        w_L_res=(1-alpha_G_tb_b)*w_res

        # Equation 26 - Mass flow rate of gas from reservoir to the well
        w_G_res=alpha_G_tb_b*w_res

        # Equation 27 - Density of gas at bottom of tubing
        rho_G_tb_b=(P_tb_b*M_G)/(R*T_tb)


        denom_alpha_b=(w_L_res*rho_G_tb_b+(w_G_inj+w_G_res)*rho_L)
        denom_alpha_b_safe=smooth_max_scaled(denom_alpha_b, 1e-9)

        # Equation 28 - Liquid volume fraction at bottom of tubing
        alpha_L_tb_b=(w_L_res*rho_G_tb_b)/denom_alpha_b_safe

        # # Equation 29 - Liquid volume fraction at top of the tubing
        alpha_L_tb_t=2*alpha_avg_L_tb-alpha_L_tb_b

        # Equation 30 - Mixture density at the top
        rho_mix_tb_t=alpha_L_tb_t*rho_L+(1-alpha_L_tb_t)*rho_G_tb_t
        rho_mix_tb_t_safe=smooth_max_scaled(rho_mix_tb_t, 1e-9)

        dP_tb_choke=P_tb_t-P_0
        # Equation 31 - Mass flow rate of mixture from choke
        w_out=K_pr*u1*sqrt(rho_mix_tb_t_safe*smooth_pos_scaled(dP_tb_choke,scale=1)+eps)

        # Equation 32 - Volumetric flow rate of production choke
        # Q_out=w_out/rho_mix_tb_t_safe

        denom_alpha_t=alpha_L_tb_t*rho_L+(1.0-alpha_L_tb_t)*rho_G_tb_t
        denom_alpha_t_safe=fmax(denom_alpha_t,1e-12)

        # Equation 33 - Gas Mass Fraction at top of tubing
        alpha_G_tb_t=((1.0-alpha_L_tb_t)*rho_G_tb_t)/denom_alpha_t_safe

        # Equation 34 - Mass flow rate of outlet gas
        w_G_out=alpha_G_tb_t*w_out

        # Equation 35 - Mass flow rate of outlet liquid
        w_L_out=(1.0-alpha_G_tb_t)*w_out

        dx1 = w_G_in - w_G_inj
        dx2 = w_G_inj + w_G_res - w_G_out
        dx3 = w_L_res - w_L_out
        dx = vertcat(dx1, dx2, dx3)

        #-----------------------
        # Algebraic outputs
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
        w_o_out = w_L_out * (1-BSW)

        V_L_tb_states=(m_L_tb/rho_L)/V_tb
        V_L_bh_states=V_bh
        alpha_L_tb=alpha_avg_L_tb
        alpha_G_tb=(1-alpha_L_tb)
        alpha_L_bh=1.0
        alpha_G_bh=0.0
        a=0
        b=0
        c=0
        d=0

        P_hidro_tb_bar=rho_avg_mix_tb*g*L_tb
        P_hidro_bh_bar=rho_avg_mix_tb*g*L_tb
        dP_int_bar=0
        w_up=w_res

        z = vertcat(
            m_G_an,
            m_G_tb,
            m_L_tb,
            a,
            b,
            c,
            d,

            # =================================
            # Hold-ups / Volumes (m³)
            # =================================
            V_L_tb_states,

            V_L_bh_states,
            # Volume fractions
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

            w_up

        )

        return dx, z

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




# def build_well_model(i: int, name_prefix="well"):
#     """
#     Returns a dict with explicit metadata + a compiled CasADi function:
#         F_all(y,u) -> (dx, out)
#     """
#     well_id = i + 1
#     fname = f"glc_well_{well_id:02d}_surrogate_casadi"
#     if fname not in globals():
#         raise AttributeError(f"No function '{fname}' in this module.")
#     well_func = globals()[fname]
#
#     # You KNOW your dimensions, so just define them once (or infer safely from a single call)
#     nx = 3
#     nu = 2
#
#     y = ca.MX.sym(f"y_{name_prefix}_{well_id}", nx)
#     u = ca.MX.sym(f"u_{name_prefix}_{well_id}", nu)
#
#     dx, out = well_func(y, u)
#
#     # compile once
#     F_all = ca.Function(
#         f"F_all_{name_prefix}_{well_id}",
#         [y, u],
#         [dx, out],
#         ["y", "u"],
#         ["dx", "out"]
#     )
#
#     return {
#         "is_dae": False,
#         "nx": nx,
#         "nu": nu,
#         "nz": 0,
#         "Z_NAMES": Z_NAMES,
#         "F_all": F_all,
#     }