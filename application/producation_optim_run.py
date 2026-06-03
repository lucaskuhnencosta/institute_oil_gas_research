"""
THIS IS THE ULTIMATE RUNNER FILE FOR THE ARTICLE AND POSSIBLE FOR THE DISSERTATION
"""
# Normalize RGB (divide by 255)
color_primary   = (54/255,  32/255, 229/255)   # blue
color_secondary = (240/255, 101/255, 74/255)   # orange
color_third     = (183/255, 53/255, 192/255)   # purple (saved)


import numpy as np
import matplotlib.pyplot as plt

from optimization.surrogate_based_optimization import SurrogateBasedOptimization
from configuration.surrogate_based_optimizer_configs import get_solver_configs
from configuration.wells import get_wells
from application.simulation_engine import make_model
from utilities.block_builders import build_casadi_surrogate_u2z_for_well, build_casadi_zero_surrogate_u2z, \
    build_casadi_polynomial_u2z_for_well
import json

def production_optim_run(
        u_guess=None,
        config=None,
        selected_wells=None,
        unconstrained_well=True,
        unconstrained_platform=True,
        P_tb_max=None,
        P_bh_min=None,
        W_max=None,
        L_max=None,
        G_available=None,
        G_max_export=None,
        surrogate_type="PINN"):
    """
    This function wraps the execution of the TRF optimization algorithm for the dissertation.
    It:
    - Can accept any well and configuration dictionaties
    - Runs Phase 1 and Phase 2 of lexicographic optimization
    - For phase 1 of lexicographic optimization, it handles the case where the problem might be infeasible
    """
    model_type = "rigorous"
    wells = get_wells()

    if config is None:
        config=get_solver_configs()

    if selected_wells is not None:
        if isinstance(selected_wells, str):
            selected_wells = [selected_wells]
        wells={
            well_name: wells[well_name] for well_name in selected_wells
        }
    well_names=list(wells.keys())

    print("=====================================================")
    print("===== RUNNING FOR THE FOLLOWING WELLS ================")
    print(well_names)
    print("=====================================================")

    rigorous_models=[]
    F_u2z_models=[]

    for well_name in well_names:
        well_data=wells[well_name]

        rigorous_models.append(make_model(
            model_type,
            BSW=well_data["BSW"],
            GOR=well_data["GOR"],
            PI=well_data["PI"],
            K_gs=well_data["K_gs"],
            K_inj=well_data["K_inj"],
            K_pr=well_data["K_pr"]
        ))
        if surrogate_type=="PINN":
            F_u2z_models.append(build_casadi_surrogate_u2z_for_well(well_name))
        elif surrogate_type=="Poly":
            F_u2z_models.append(build_casadi_polynomial_u2z_for_well(well_name))
        elif surrogate_type=="zero":
            F_u2z_models.append(build_casadi_zero_surrogate_u2z())
        else:
            raise ValueError(f"Unknown surrogate_type: {surrogate_type}")

    try:
        opt1=SurrogateBasedOptimization(config=config,
                                        wells=wells,
                                        rigorous_models=rigorous_models,
                                        F_u2z_models=F_u2z_models,
                                        unconstrained_well=unconstrained_well,
                                        unconstrained_platform=unconstrained_platform,
                                        P_tb_max=P_tb_max,
                                        P_bh_min=P_bh_min,
                                        W_max=W_max,
                                        L_max=L_max,
                                        G_available=G_available,
                                        G_max_export=G_max_export,
                                        refinement=False,
                                        u_guess=u_guess,
                                        total_wo=None)
        result1=opt1.solve()
    except Exception as e:
        return {
            "success": False,
            "failed_phase": "phase_1_exception",
            "error": str(e),
        }, None, None, None, None, None

    if not result1.get("success", False):
        return None, None, None, None, None, None

    u_phase1 = result1["u_opt"]
    total_wo_phase1 = -result1["phi_opt"]

    print(f"total_wo_phase1: {total_wo_phase1}")

    try:
        opt2 = SurrogateBasedOptimization(
            config=config,
            wells=wells,
            rigorous_models=rigorous_models,
            F_u2z_models=F_u2z_models,
            unconstrained_well=unconstrained_well,
            unconstrained_platform=unconstrained_platform,
            P_tb_max=P_tb_max,
            P_bh_min=P_bh_min,
            W_max=W_max,
            L_max=L_max,
            G_available=G_available,
            G_max_export=G_max_export,
            refinement=True,
            u_guess=u_phase1,
            total_wo=total_wo_phase1,
        )
        result2 = opt2.solve()
    except Exception as e:
        result2 = {
            "success": False,
            "failed_phase": "phase_2_exception",
            "error": str(e),
        }
        return result1, result2, total_wo_phase1, u_phase1, None, None

    if not result2.get("success", False):
        return result1, result2, total_wo_phase1, u_phase1, None, None

    total_gas_inj=result2["phi_opt"]
    u_final=result2["u_opt"]

    return result1, result2, total_wo_phase1, u_phase1, total_gas_inj, u_final