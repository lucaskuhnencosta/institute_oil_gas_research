import numpy as np

def get_solver_configs():
    return {
        "out_names":[ "P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res", "w_L_res", "w_G_res", "w_w_out","w_o_out"],
        "u_lb": [0.05,0.10],
        "u_ub": [1.00,1.00],
        "u_min": [0.05,0.10],
        "u_max": [1.00,1.00],
        # global constraints
        "P_min_bh":90,
        "P_max_tb_b":120,
        "G_available":14.00,
        "G_max_export":1.40,
        "W_max":11.50,
        "L_max":40.0,
        "unconstrained_well":False,
        "unconstrained_platform":False,
        "enforce_stable":True,
        "Delta":1.0,
        "gamma_c":0.4,
        "gamma_e":2.0,
        "gamma_f":0.01,
        "gamma_theta":0.01,
        "k_theta":0.01,
        "gamma_s":0.9,
        "eta_1":0.4,
        "eta_2":0.8,
        "theta_tol":1.0e-4,
        "u_guess_list": np.array([[0.13385,0.4038],[0.31307,0.5000]]),
        "max_iter": 100}
    ## need to undestans
