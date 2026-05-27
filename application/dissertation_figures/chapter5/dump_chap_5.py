from application.producation_optim_run import production_optim_run
import numpy as np

result1 = production_optim_run(
    u_guess=np.array([[1.00, 1.00]]),
    config=None,
    selected_wells=["P1"],
)

print(result1)