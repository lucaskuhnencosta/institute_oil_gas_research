from configuration.scenarios import get_scenarios
from configuration.wells import get_wells
from optimization.production_optimizer_NN import optimize_field_production_MODEL
from utilities.block_builders import print_solution, print_compact_summary, polyval_casadi
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
from solvers.steady_state_solver import solve_equilibrium_ipopt
import pandas as pd
import numpy as np
import casadi as ca

# print_solution(sol_rigorous, show_states=False, show_algebraic=True)
# print_compact_summary(sol_rigorous)

wells=get_wells()
model_type="rigorous"
u_guess_list=[[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]]

from application.simulation_engine import make_model, dm_to_float

constraint_scenarios=get_scenarios()
from optimization.production_optimizer import optimize_field_production

results_rows_plant=[]


def validate_optimization_solution_on_plant(
    sol_model,
):

    per_well_validated = []
    totals = {
        "w_o_out": 0.0,
        "w_w_out": 0.0,
        "w_G_inj": 0.0,
        "w_G_out": 0.0,
        "w_L_out": 0.0,
    }
    plant_models = {}

    for well_name, well_data in wells.items():
        plant_models[well_name] = make_model(
            model_type,
            BSW=well_data["BSW"],
            GOR=well_data["GOR"],
            PI=well_data["PI"],
            K_gs=well_data["K_gs"],
            K_inj=well_data["K_inj"],
            K_pr=well_data["K_pr"]
        )

    for well_result in sol_model["per_well"]:
        well_name = well_result["well_name"]
        # Optimal u recommended by PINN or Polynomial optimizer
        u_opt = well_result["u"]
        model = plant_models[well_name]
        Z_NAMES = model["Z_NAMES"]

        y_guess = wells[well_name]["y_guess_rig"]
        z_guess = wells[well_name]["z_guess_rig"]

        # Evaluate the surrogate-recommended u in the rigorous plant model
        if model["is_dae"]:
            y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                model=model,
                u_val=u_opt,
                y_guess=y_guess,
                z_guess=z_guess,
            )
        else:
            y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                model=model,
                u_val=u_opt,
                y_guess=y_guess,
            )
            z_star = None
            g_star = None

        # Convert output vector to dictionary
        out_dict = {
            name: dm_to_float(out_star[i])
            for i, name in enumerate(Z_NAMES)
        }

        # Add to field totals
        totals["w_o_out"] += out_dict["w_o_out"]
        totals["w_w_out"] += out_dict["w_w_out"]
        totals["w_G_inj"] += out_dict["w_G_inj"]
        totals["w_G_out"] += out_dict["w_G_out"]
        totals["w_L_out"] += out_dict["w_L_out"]

        per_well_validated.append({
            "well_name": well_name,
            "u": np.array(u_opt, dtype=float).reshape(-1),
            "y": y_star,
            "z": z_star,
            "dx": dx_star,
            "g": g_star,
            "out": out_star,
            "out_dict": out_dict,
            "stable": stable,
            "stats": stats,
            "P_bh_bar": out_dict["P_bh_bar"],
            "P_tb_b_bar": out_dict["P_tb_b_bar"],
        })

    return {
        "totals": totals,
        "per_well": per_well_validated,
    }

def make_validation_row(sim_type, scenario_code, validated_result):
    """
    Creates one dataframe row from plant validation result.
    """

    row = {
        "sim_type": sim_type,
        "scenario": scenario_code,
    }

    # Add field totals
    for key, value in validated_result["totals"].items():
        row[key] = value

    # Add well pressures
    for well_result in validated_result["per_well"]:
        well_name = well_result["well_name"]

        row[f"P_bh_bar_{well_name}"] = well_result["P_bh_bar"]
        row[f"P_tb_b_bar_{well_name}"] = well_result["P_tb_b_bar"]

    return row


rows=[]

for case_code, case in constraint_scenarios.items():
    unconstrained_well=case["unconstrained_well"]
    unconstrained_platform=case["unconstrained_platform"]
    P_tb_max = case["P_tb_max"]
    P_bh_min = case["P_bh_min"]
    W_max = case["W_max"]
    L_max = case["L_max"]
    G_available = case["G_available"]
    G_max_export = case["G_max_export"]


    sol_rigorous = optimize_field_production(
        model_type=model_type,
        wells=wells,
        u_guess_list=u_guess_list,
        G_available=G_available,
        G_max_export=G_max_export,
        W_max=W_max,
        L_max=L_max,
        unconstrained_well=unconstrained_well,
        unconstrained_platform=unconstrained_platform,
        enforce_stable=True
    )
    val_rigorous = validate_optimization_solution_on_plant(
        sol_model=sol_rigorous,
    )
    rows.append(
        make_validation_row(
            sim_type="Rigorous",
            scenario_code=case_code,
            validated_result=val_rigorous,
        ))


    #=====================================================================
    #============== PINN =================================================
    # =====================================================================

    sol_PINN = optimize_field_production_MODEL(
        wells=wells,
        u_guess_list=u_guess_list,
        G_available=G_available,
        G_max_export=G_max_export,
        W_max=W_max,
        L_max=L_max,
        unconstrained_well=unconstrained_well,
        unconstrained_platform=unconstrained_platform,
        enforce_stable=True,
    )
    val_pinn = validate_optimization_solution_on_plant(
        sol_model=sol_PINN,
    )
    rows.append(
        make_validation_row(
            sim_type="PINN_validated_on_plant",
            scenario_code=case_code,
            validated_result=val_pinn,
        ))


    #=====================================================================
    #============== Poly =================================================
    # =====================================================================


    sol_poly = optimize_field_production_MODEL(
        wells=wells,
        u_guess_list=u_guess_list,
        G_available=G_available,
        G_max_export=G_max_export,
        W_max=W_max,
        L_max=L_max,
        unconstrained_well=unconstrained_well,
        unconstrained_platform=unconstrained_platform,
        enforce_stable=True,
        is_poly=True
    )
    val_poly = validate_optimization_solution_on_plant(
        sol_model=sol_poly,
    )
    rows.append(
        make_validation_row(
            sim_type="Poly_validated_on_plant",
            scenario_code=case_code,
            validated_result=val_poly,
        ))



    # results_rows.append(row)


df_validated = pd.DataFrame(rows)
print(df_validated)