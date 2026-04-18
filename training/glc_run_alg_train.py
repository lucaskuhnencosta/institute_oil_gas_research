import torch
import torch.nn as nn

from networks.networks import AlgNN
from training.glc_alg_train_new_version import AlgTrainer

HIDDEN_UNITS = [64] * 4


# 3. training Parameters
N_TRAIN_SAMPLES = 40000  # Use a large dataset for good accuracy
N_VAL_SAMPLES = 1089
EPOCHS = 10000
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# 4. System & Logging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_PROJECT = "PINC-GasLift-AlgNN"
WANDB_GROUP = "AlgNN_Run_1"

alg_net = AlgNN(
    hidden_units=HIDDEN_UNITS,
    Nonlin=nn.Tanh
).to(DEVICE)


trainer = AlgTrainer(
    net=alg_net,
    N_train=N_TRAIN_SAMPLES,
    N_val=N_VAL_SAMPLES,
    lr=LEARNING_RATE,
    mixed_precision=True if DEVICE == "cuda" else False,
    device=DEVICE,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    random_seed=RANDOM_SEED
)

trainer.run()