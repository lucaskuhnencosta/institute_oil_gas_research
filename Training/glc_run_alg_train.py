import torch
import torch.nn as nn


from PINC.core.networks import AlgNN
from PINC.experiments.Gas_Lift.glc_alg_trainer import AlgTrainer

HIDDEN_UNITS = [30] * 4

# 2. Normalization Bounds (from friend's 'znet_well_1_single' function)
# Input Y (States)
Y_MIN_ALG = [3200.0, 180.0, 7000.0]
Y_MAX_ALG = [4700.0, 400.0, 11000.0]
# Input U (Controls)
U_MIN_ALG = [0.0, 0.0]
U_MAX_ALG = [1.0, 1.0]
# Output Z (Algebraic)
Z_MIN_ALG = [70.0, 0.0, 0.0, 0.0]  # [Pbh, w_G_in, w_G_res, w_L_res]
Z_MAX_ALG = [145.0, 2.0, 12.0, 100.0]  # [Pbh, w_G_in, w_G_res, w_L_res]

# 3. Training Parameters
N_TRAIN_SAMPLES = 100000  # Use a large dataset for good accuracy
N_VAL_SAMPLES = 5000
EPOCHS = 10000
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# 4. System & Logging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_PROJECT = "PINC-GasLift-AlgNN"
WANDB_GROUP = "AlgNN_Run_1"

alg_net = AlgNN(
    hidden_units=HIDDEN_UNITS,
    y_min=Y_MIN_ALG,
    y_max=Y_MAX_ALG,
    u_min=U_MIN_ALG,
    u_max=U_MAX_ALG,
    z_min=Z_MIN_ALG,
    z_max=Z_MAX_ALG,
    Nonlin=nn.Tanh
).to(DEVICE)


trainer = AlgTrainer(
    net=alg_net,
    N_train=N_TRAIN_SAMPLES,
    N_val=N_VAL_SAMPLES,
    lr=LEARNING_RATE,
    y_min_train=Y_MIN_ALG,
    y_max_train=Y_MAX_ALG,
    u_min_train=U_MIN_ALG,
    u_max_train=U_MAX_ALG,
    mixed_precision=True if DEVICE == "cuda" else False,
    device=DEVICE,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    random_seed=RANDOM_SEED
)

trainer.run()