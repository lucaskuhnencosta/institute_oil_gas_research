import torch

from networks.networks import PINN
from simulators.surrogate_simulator.surrogate_model_torch import glc_surrogate_dx_torch

# Import your patched trainer
from training.glc_pinn_train import SteadyStatePINNTrainer

# Import well lists
from configuration.wells import get_wells
from settings import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

wells= get_wells()
well_name="P6"
well_list=wells[well_name]

# 1) Build vanilla net: u -> y
net = PINN(hidden_units=[64,64,64],
           n_u=2,
           n_y=3,
           y_min=well_list["y_min"],
           y_max=well_list["y_max"],
           u_min=[U1_MIN, U2_MIN],
           u_max=[U1_MAX, U2_MAX],
           )


# 2) Trainer
trainer = SteadyStatePINNTrainer(
    net=net,
    surrogate_function=glc_surrogate_dx_torch,
    well_list=well_list,
    well_name=well_name,
    N_col=1000,
    lambda_data=1,
    lambda_physics=1e-4,
    adam_epochs_1=10000,
    adam_epochs_2=10000,
    lbfgs_epochs=2000,
    lr=1e-3,
    mse_f_scale_factors=[1.0, 1.0, 1.0],
    wandb_project=well_name,
    random_seed=333333)

results = trainer.run()
print("Done.")
print("Best val:", results["best_val_loss"])
print("Train at best val:", results["train_loss_at_best_val"])

