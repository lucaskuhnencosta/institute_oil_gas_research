import torch

from networks.networks import PINN
from simulators.surrogate_simulator.surrogate_model_torch import glc_surrogate_dx_torch_nostuck

# Import your patched trainer
from training.glc_pinn_trainer import SteadyStatePINNTrainer

# Import well lists
from configuration.wells import get_wells

device = "cuda:0" if torch.cuda.is_available() else "cpu"

wells= get_wells()
well_name="P2"
well_list=wells[well_name]

# 1) Build vanilla net: u -> y
net = PINN(hidden_units=[64, 64, 64],
           n_u=2,
           n_y=3,
           y_min=well_list["y_min"],
           y_max=well_list["y_max"],
           u_min=[0.05, 0.10],
           u_max=[1.0, 1.0],
           )


# 2) Trainer
trainer = SteadyStatePINNTrainer(
    net=net,
    surrogate_function=glc_surrogate_dx_torch_nostuck,
    well_list=well_list,
    N_col=5000,
    N_data=30,
    lambda_data=1000,
    adam_epochs=12000,
    lbfgs_epochs=2000,
    lr=1e-3,
    u_min=[0.05, 0.10],
    u_max=[1.0, 1.0],
    mse_f_scale_factors=[1.0, 1.0, 1.0],
    u_probe=(0.6, 0.6),
    wandb_project=well_name,
    random_seed=333333)

results = trainer.run()
print("Done.")
print("Best val:", results["best_val_loss"])
print("Train at best val:", results["train_loss_at_best_val"])

