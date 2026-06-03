from application.producation_optim_run import production_optim_run
from configuration.wells import get_wells
from configuration.scenarios import get_scenarios
import numpy as np
from optimization.production_optimizer import optimize_field_production
scenarios=get_scenarios()
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

from solvers.steady_state_solver import solve_equilibrium_ipopt
from application.simulation_engine import make_model, dm_to_float, validate_optimization_solution_on_plant




final_results=[]
all_runs = []

for case_code, case in scenarios.items():
    unconstrained_well=case["unconstrained_well"]
    unconstrained_platform=case["unconstrained_platform"]
    P_tb_max = case["P_tb_max"]
    P_bh_min = case["P_bh_min"]
    W_max = case["W_max"]
    L_max = case["L_max"]
    G_available = case["G_available"]
    G_max_export = case["G_max_export"]

    cols = [
        "phase",
        "iteration",
        "debug_corrected",
        "Delta",
        "accepted",
        "rejected",
        "rejected_reason",
        "theta_k",
        "phi_k",
        "type"
    ]


    print("\n\n\n")
    print("==============================================================================================")
    print("==============================================================================================")
    print("==============================================================================================")
    print(f"====================   RUNNING PROBLEM OF CODE {case_code}   ================================")
    print("==============================================================================================")
    print("==============================================================================================")
    print("==============================================================================================")
    print("\n")

    #########################################################################################################
    # FIRST WE SOLVE USING THE PROFESSIONAL SOLVER
    #########################################################################################################
    model_type = "rigorous"
    wells = get_wells()

    sol_rigorous = optimize_field_production(
        model_type=model_type,
        wells=wells,
        u_guess_list=[[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]],
        G_available=G_available,
        G_max_export=G_max_export,
        W_max=W_max,
        L_max=L_max,
        unconstrained_well=unconstrained_well,
        unconstrained_platform=unconstrained_platform,
        enforce_stable=True
    )
    total_oil = float(sol_rigorous["totals"]["w_o_out"])
    total_g_inj = float(sol_rigorous["totals"]["w_G_inj"])

    #########################################################################################################
    # NOW WE SOLVE USING OUR SOLVER
    #########################################################################################################

    result1, result2, total_wo_phase1, u_phase1, total_gas_inj, u_final=production_optim_run(
            u_guess=None,
            config=None,
            selected_wells=None,
            unconstrained_well=unconstrained_well,
            unconstrained_platform=unconstrained_platform,
            P_tb_max=P_tb_max,
            P_bh_min=P_bh_min,
            W_max=W_max,
            L_max=L_max,
            G_available=G_available,
            G_max_export=G_max_export,
            surrogate_type="zero")

    #########################################################################################################
    # STORING SOLVER HISTORY
    #########################################################################################################
    if result1 is not None:
        df_history_1 = pd.DataFrame(result1["history"])
        df_view_1 = df_history_1[cols].copy()
        n_hist_obj_1=len(df_history_1)

    if result2 is not None:
        df_history_2=pd.DataFrame(result2["history"])
        df_view_2 = df_history_2[cols].copy()
        n_hist_obj_2=len(df_history_2)

        df_view= pd.concat([df_view_1, df_view_2], ignore_index=False)
    else:
        df_view = df_view_1.copy()
        n_hist_obj_2 = 0

    df_view["code"] = case_code
    df_view = df_view[["code"] + cols]

    all_runs.append(df_view)

    #########################################################################################################
    # INFERING TO CHECK RESULTS
    #########################################################################################################
    if result1 is not None:
        u_phase1_pairs=u_phase1.reshape(6,2)
        proof_that_works_interm=validate_optimization_solution_on_plant(u_phase1_pairs)
        u_guess_list_phase1 = u_phase1_pairs

    if result2 is not None:
        u_pairs= u_final.reshape(6, 2)
        proof_that_works = validate_optimization_solution_on_plant(u_pairs)
        u_guess_list = u_pairs

    #########################################################################################################
    # FINAL NUMBER OF INTEREST
    #########################################################################################################
    if result1 is not None:
        phase_1_oil_check = proof_that_works_interm["totals"]["w_o_out"]
        phase_1_gas_check = proof_that_works_interm["totals"]["w_G_inj"]
    else:
        phase_1_oil_check = None
        phase_1_gas_check = None


    if result2 is not None:
        phase_2_oil_check = proof_that_works["totals"]["w_o_out"]
        phase_2_gas_check = proof_that_works["totals"]["w_G_inj"]
    else:
        phase_2_oil_check = None
        phase_2_gas_check = None

    if result1 is not None and result2 is not None:
        reduction_oil = (1 - phase_2_oil_check / phase_1_oil_check) * 100
        reduction_gas = (1 - phase_2_gas_check / phase_1_gas_check) * 100
        reduction_gas_bmk = (1 - phase_2_gas_check / total_g_inj) * 100
    else:
        reduction_oil = None
        reduction_gas = None
        reduction_gas_bmk = None



    final_results.append({
        "case_code": case_code,
        # Just normally solving
        "optimal_oil":total_oil,
        "gas_non_opt":total_g_inj,
        # Phase 1 and corresponding gas
        "phase_1_oil": total_wo_phase1,
        "phase_1_gas_check": phase_1_gas_check,
        # Checking phase 1 and corresponding gas
        "phase_1_oil_check": phase_1_oil_check,
        # Phase 2 and corresponding gas
        "phase_2_gas": total_gas_inj,
        "phase_2_oil_check": phase_2_oil_check,
        "phase_2_gas_check": phase_2_gas_check,
        "Reduction oil (%)":reduction_oil,
        "Reduction gas (%)":reduction_gas,
        "Reduction gas bmk (%)":reduction_gas_bmk,
        "n_obj1": n_hist_obj_1,
        "n_obj2": n_hist_obj_2,
    })

    print(df_view)

print('\n\n')

df_final_results=pd.DataFrame(final_results)

df_all_runs = pd.concat(all_runs, ignore_index=True)

print("\n\n")
print("===================================================")
print("================   ALL RUNS   ====================")
print("===================================================")
print(df_all_runs)

df_all_runs.to_csv("lexicographic_all_runs.csv", index=False)

print("\n\n")
print("===================================================")
print("================   FINAL RESULTS   ====================")
print("===================================================")
with pd.option_context("display.float_format", "{:.2f}".format):
    print(df_final_results)

df_final_results.to_csv("lexicographic_final_results.csv", index=False)