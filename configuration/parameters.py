import numpy as np

def get_parameters():
    """
    These are the final global parameters used in the thesis
    """
    return{
    # Universal constants
    "g": 9.81,
    "R": 8.314,
    # Annulus
    "D_an":0.20,
    "A_an":0.031415926535897934,
    "L_an":2048.0,
    "V_an":64.33981754551897,
    # Bottom Section
    "D_bh":0.20,
    "A_bh":0.031415926535897934,
    "L_bh":75.0,
    "V_bh":2.356194490192345,
    # Tubing
    "D_tb":0.127,
    "A_tb": 0.012667686977437444,
    "L_tb":1973.0,
    "V_tb":24.99334640648408,
    # Densities
    "rho_o":760,
    "rho_w":1000,
    # Viscosities
    "mu_o": 3.64e-3,
    "mu_w": 1.00e-3,
    # Temperatures
    "T_an": 348,
    "T_bh": 371.5,
    "T_tb": 369.4,
    "M_G": 0.0167,
    # Global parameters
    "P_gs": 140e5,
    "P_res": 160e5,
    "P_0": 20e5,
    #Friction
    "epsilon_tubing": 3e-4,
    "K0_int": 2.50
    }