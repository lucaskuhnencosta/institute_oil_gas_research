import numpy as np

def get_solver_configs():
    return {
        "debug_verbose": False,
        "theta_compat_tol": 1e-4,
        "out_names":[ "P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res", "w_L_res", "w_G_res", "w_w_out","w_o_out"],
        # global constraints
        "enforce_stable":True,
        "warm_start":False,
        "oil_allowance":0.001,#0.001 = 0.10%
        "Delta_restoration":1.00,
        "gamma_e_restoration":1.01 ,
        "gamma_c_restoration":0.99 ,
        "Delta":1.00,
        "gamma_c":0.90,
        "gamma_e":1.8,
        "gamma_f":0.01,
        "gamma_theta":0.01,
        "k_theta":0.01,
        "gamma_s":0.9,
        "eta_1":0.4,
        "eta_2":0.8,
        "theta_tol":2.0e-4,
        "u_guess_list":np.array([[0.50, 0.50], [0.50, 0.50],[0.50, 0.50], [0.50, 0.50],[0.50, 0.50],[0.50, 0.50]]),
        "max_iter": 5000}