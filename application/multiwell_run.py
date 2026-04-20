"""
THIS IS THE ULTIMATE RUNNER FILE FOR THE ARTICLE AND POSSIBLE FOR THE DISSERTATION
"""

import numpy as np

from optimization.surrogate_based_optimization import SurrogateBasedOptimization
from configuration.surrogate_based_optimizer_configs import get_solver_configs
from configuration.wells import get_wells
from application.simulation_engine import make_model
from utilities.block_builders import build_casadi_surrogate_u2z_for_well

wells=get_wells()
config=get_solver_configs()
model_type = "rigorous"

well_names = list(wells.keys())
N = len(well_names)

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
    F_u2z_models.append(build_casadi_surrogate_u2z_for_well(well_name))

opt=SurrogateBasedOptimization(config=config,
                               wells=wells,
                               rigorous_models=rigorous_models,
                               F_u2z_models=F_u2z_models)

result=opt.solve()