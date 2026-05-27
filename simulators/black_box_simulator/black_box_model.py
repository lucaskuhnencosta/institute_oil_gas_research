"""
This simulator is in its final version for the dissertation and adequately described in Appendix C
No further changes are needed.
"""

from casadi import *
import numpy as np
import casadi as ca
from configuration.parameters import get_parameters

parameters = get_parameters()

def make_glc_well_rigorous(BSW,
                           GOR,
                           PI,
                           K_gs,
                           K_inj,
                           K_pr):

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

        # Annulus
        V_an = parameters["V_an"]
        L_an = parameters["L_an"]

        # Bottom section
        D_bh=parameters["D_bh"]
        L_bh=parameters["L_bh"]
        S_bh=parameters["A_bh"]
        V_bh =parameters["V_bh"]

        # Tubing
        L_tb = parameters["L_tb"]
        D_tb = parameters["D_tb"]
        S_tb = parameters["A_tb"]
        V_tb =parameters["V_tb"]

        R = parameters["R"]
        g = parameters["g"]

        mu_o = parameters["mu_o"]
        mu_w = parameters["mu_w"]
        rho_o =parameters["rho_o"]
        rho_w =parameters["rho_w"]

        T_an = parameters["T_an"]
        T_bh = parameters["T_bh"]
        T_tb = parameters["T_tb"]
        M_G = parameters["M_G"]

        # Pressures
        P_gs = parameters["P_gs"]
        P_res = parameters["P_res"]
        P_0 = parameters["P_0"]

        # Friction
        K0_int=parameters["K0_int"]
        epsilon_tubing = parameters["epsilon_tubing"]




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

        alpha_L_tb=V_L_tb_states/V_tb
        alpha_G_tb=1.0-alpha_L_tb

        alpha_L_bh=V_L_bh_states/V_bh
        alpha_G_bh=1.0-alpha_L_bh

        # Mixture density in each CV (volume averaged)
        rho_mix_tb = (m_o_t+m_w_t+m_G_t)/V_tb
        rho_mix_bh = (m_o_b+m_w_b+m_G_b)/V_bh
        rho_mix_tb_safe = smooth_max_scaled(rho_mix_tb, 1e-9)
        rho_mix_bh_safe = smooth_max_scaled(rho_mix_bh, 1e-9)

        # -----------------------
        # PART 5 - Decomposition of w_up
        # -----------------------

        # Definitions of mass fractions
        # Bottom hole

        m_tot_bh = m_G_b + m_o_b + m_w_b
        m_L_bh = m_o_b + m_w_b

        m_tot_bh_safe = smooth_max_scaled(m_tot_bh, 1e-9)
        m_L_bh_safe = smooth_max_scaled(m_L_bh, 1e-9)

        x_G_bh = m_G_b / m_tot_bh_safe
        x_w_bh = m_w_b / m_L_bh_safe

        # Tubing

        m_tot_tb = m_G_t + m_o_t + m_w_t
        m_L_tb = m_o_t + m_w_t

        m_tot_tb_safe = smooth_max_scaled(m_tot_tb, 1e-9)
        m_L_tb_safe = smooth_max_scaled(m_L_tb, 1e-9)

        x_G_tb = m_G_t / m_tot_tb_safe
        x_w_tb = m_w_t / m_L_tb_safe

        ###
        w_G_up = x_G_bh * w_up_g
        w_L_up = (1.0 - x_G_bh) * w_up_g

        w_w_up = x_w_bh * w_L_up
        w_o_up = (1.0 - x_w_bh) * w_L_up

        # -----------------------
        # PART 6 - Superficial velocities
        # -----------------------

        # Superficial velocities bottom
        U_sg_b=w_G_res/(rho_G_bh*S_bh) # 4.1
        # U_sl_b=w_L_res/(rho_L*S_bh) # 4.2
        rho_L_bh=1/((x_w_bh/rho_w)+((1-x_w_bh)/rho_o))
        U_sl_b=w_L_res/(rho_L_bh*S_bh)

        # Superficial velocities top
        U_sg_t=(w_G_up+w_G_inj)/(rho_G_tb*S_tb) # 4.3
        # U_sl_t=(w_L_up)/(rho_L*S_tb) # 4.4
        rho_L_tb=1/((x_w_tb/rho_w)+((1-x_w_tb)/rho_o))
        U_sl_t = (w_L_up) / (rho_L_tb * S_tb)

        U_avg_b=U_sg_b+U_sl_b # 4.5
        U_avg_t=U_sg_t+U_sl_t # 4.6

        # -----------------------
        # PART 7 - Friction and gravity
        # -----------------------

        mu_tb =np.exp((1 - x_w_tb) * np.log(mu_o) + x_w_tb * np.log(mu_w))
        mu_bh=np.exp((1 - x_w_bh) * np.log(mu_o) + x_w_bh * np.log(mu_w))
        # In the top section
        Re_tb = rho_mix_tb_safe*U_avg_t*D_tb/mu_tb
        Re_tb_safe=smooth_max_scaled(Re_tb,1.0)  # keep >0 for 6.9/Re

        log_arg_tb = (epsilon_tubing / (D_tb * 3.7)) ** 1.11 + 6.9 / Re_tb_safe
        log_arg_tb_safe = fmax(log_arg_tb, 1e-12)
        lambda_tb = (1 / (1.8 * log10(log_arg_tb_safe))) ** 2

        F_t = (alpha_L_tb*lambda_tb*rho_mix_tb*U_avg_t**2*L_tb)/(2*D_tb)

        dP_t=rho_mix_tb*g*L_tb+F_t
        P_tb_b=P_tb_t+dP_t

        g1=P_tb_b-P_tb_b_g # Here we close the loop for the tubing bottom pressure

        # In the bottom section
        Re_bh = rho_mix_bh_safe*U_avg_b*D_bh/mu_bh
        Re_bh_safe=smooth_max_scaled(Re_bh, 1.0)

        log_arg_bh=(epsilon_tubing/(D_bh*3.7))**1.11+6.9/Re_bh_safe
        log_arg_bh_safe=fmax(log_arg_bh, 1e-12)
        lambda_bh=(1.0/(1.8*log10(log_arg_bh_safe)))**2

        F_bh=(alpha_L_bh*lambda_bh*rho_mix_bh*U_avg_b**2*L_bh)/(2.0*D_bh)

        dP_bh_seg=rho_mix_bh*g*L_bh+F_bh
        P_bh=P_bh_t+dP_bh_seg

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

        denom_alpha_b=(w_L_up*rho_G_tb_b+(w_G_inj+w_G_up)*rho_L_bh)
        denom_alpha_b_safe=smooth_max_scaled(denom_alpha_b, 1e-9)

        alpha_L_tb_b=w_L_up*rho_G_tb_b/denom_alpha_b_safe
        alpha_L_tb_t = 2 * alpha_L_tb - alpha_L_tb_b

        # Equation 30 - Mixture density at the top
        rho_mix_tb_t = alpha_L_tb_t * rho_L_tb + (1 - alpha_L_tb_t) * rho_G_tb
        rho_mix_tb_t_safe = smooth_max_scaled(rho_mix_tb_t, 1e-9)

        dP_tb_choke = P_tb_t - P_0
        w_out = K_pr * u1 * ca.sqrt(rho_mix_tb_t_safe * smooth_pos_scaled(dP_tb_choke) + eps)

        denom_alpha_t = alpha_L_tb_t * rho_L_tb + (1.0 - alpha_L_tb_t) * rho_G_tb
        denom_alpha_t_safe = fmax(denom_alpha_t, 1e-12)

        # Equation 33 - Gas Mass Fraction at top of tubing
        alpha_G_tb_t = ((1.0 - alpha_L_tb_t) * rho_G_tb) / denom_alpha_t_safe

        w_G_out = alpha_G_tb_t * w_out

        w_L_out = (1.0 - alpha_G_tb_t) * w_out

        # Liquid mass fraction of water in tubing
        Y_w_L_t = m_w_t/smooth_max_scaled(m_w_t + m_o_t, 1e-9)
        Y_w_L_t = ca.fmin(ca.fmax(Y_w_L_t, 0.0), 1.0)

        w_w_out = Y_w_L_t * w_L_out
        w_o_out = (1.0 - Y_w_L_t) * w_L_out

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
        P_an_t_bar=P_an_t / 1e5
        P_an_b_bar=P_an_b / 1e5

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
            P_an_t_bar,
            P_an_b_bar,
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

            Re_tb,
            Re_bh,
            U_avg_t,
            U_avg_b,

            # =================================
            # Flows (kg/s)
            # =================================
            w_res,
            w_L_res,
            w_G_res,

            w_out,
            w_G_out,
            w_L_out,
            w_w_out,
            w_o_out,

            w_G_inj,
            w_up_g,
            rho_mix_tb,
        )
        return dx, alg, out

    return glc_well




