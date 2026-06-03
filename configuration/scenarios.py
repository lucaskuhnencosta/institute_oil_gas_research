def get_scenarios():
    return {
    "S0": {
        "unconstrained_well": True,
        "unconstrained_platform": True,
        "P_tb_max": None,
        "P_bh_min": None,
        "W_max": None,
        "L_max": None,
        "G_available": None,
        "G_max_export": None,
    },

    "S1": {
        "unconstrained_well": False,
        "unconstrained_platform": True,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": None,
        "L_max": None,
        "G_available": None,
        "G_max_export": None,
    },

    "S2": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 18.0,
        "L_max": 135.0,
        "G_available": 55.0,
        "G_max_export": 3.70,
    },

    "S3": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 18.0,
        "L_max": 120.0,
        "G_available": 55.00,
        "G_max_export": 3.70,
    },

    "S4": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 18.0,
        "L_max": 120.0,
        "G_available": 40.00,
        "G_max_export": 3.00,
    },

    "S5": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 18.0,
        "L_max": 120.0,
        "G_available": 40.00,
        "G_max_export": 3.00,
    },

    "S6": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 15.0,
        "L_max": 120.0,
        "G_available": 40.00,
        "G_max_export": 3.00,
    },

    "S7": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 15.0,
        "L_max": 80.0,
        "G_available": 40.0,
        "G_max_export": 3.00,
    },

    "S8": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 15.0,
        "L_max": 80.0,
        "G_available": 35,
        "G_max_export": 3.00,
    },

    "S9": {
        "unconstrained_well": False,
        "unconstrained_platform": False,
        "P_tb_max": 120.0,
        "P_bh_min": 80.0,
        "W_max": 15.0,
        "L_max": 80.0,
        "G_available": 35.0,
        "G_max_export": 2.00,
    },
}