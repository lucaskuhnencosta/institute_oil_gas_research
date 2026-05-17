from application.prod_optim_run import prod_optim_run
from configuration.surrogate_based_optimizer_configs import get_solver_configs
from configuration.wells import get_wells

wells=get_wells()
config=get_solver_configs()

result_1,result_2=prod_optim_run(
        selected_wells="P1")