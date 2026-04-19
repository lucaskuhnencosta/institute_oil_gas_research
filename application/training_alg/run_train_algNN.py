import torch
import torch.nn as nn

from networks.networks import AlgNN
from training.glc_alg_train import AlgTrainer
from configuration.wells import get_wells

wells = get_wells()

well_name="P1"
well_list=wells[well_name]

HIDDEN_UNITS = [64] * 4

# 3. training Parameters
N_TRAIN_SAMPLES = 80 # Use a large dataset for good accuracy
N_VAL_SAMPLES = 4
EPOCHS = 10000
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# 4. System & Logging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_PROJECT = "PINC-GasLift-AlgNN"
WANDB_GROUP = "AlgNN_Run_1"

alg_net = AlgNN(
    hidden_units=HIDDEN_UNITS,
    Nonlin=nn.Tanh,
    y_min=well_list["y_min"],
    y_max=well_list["y_max"],
    z_min=well_list["z_min"],
    z_max=well_list["z_max"],
).to(DEVICE)


trainer = AlgTrainer(
    net=alg_net,
    well_list=well_list,
    N_train=N_TRAIN_SAMPLES,
    N_val=N_VAL_SAMPLES,
    adam_epochs=10000,
    lbfgs_epochs=10000,
    u_min=[0.05, 0.10],
    u_max=[1.0, 1.0],
    lr=LEARNING_RATE,
    mixed_precision=True if DEVICE == "cuda" else False,
    device=DEVICE,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    random_seed=RANDOM_SEED
)

trainer.run()


well_name="P2"
well_list=wells[well_name]

HIDDEN_UNITS = [64] * 4

# 3. training Parameters
N_TRAIN_SAMPLES = 80 # Use a large dataset for good accuracy
N_VAL_SAMPLES = 4
EPOCHS = 10000
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# 4. System & Logging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_PROJECT = "PINC-GasLift-AlgNN"
WANDB_GROUP = "AlgNN_Run_1"

alg_net = AlgNN(
    hidden_units=HIDDEN_UNITS,
    Nonlin=nn.Tanh,
    y_min=well_list["y_min"],
    y_max=well_list["y_max"],
    z_min=well_list["z_min"],
    z_max=well_list["z_max"],
).to(DEVICE)


trainer = AlgTrainer(
    net=alg_net,
    well_list=well_list,
    N_train=N_TRAIN_SAMPLES,
    N_val=N_VAL_SAMPLES,
    adam_epochs=10000,
    lbfgs_epochs=10000,
    u_min=[0.05, 0.10],
    u_max=[1.0, 1.0],
    lr=LEARNING_RATE,
    mixed_precision=True if DEVICE == "cuda" else False,
    device=DEVICE,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    random_seed=RANDOM_SEED
)

trainer.run()