import torch

from networks.networks import PINN
from simulators.surrogate_simulator.surrogate_model_torch import glc_surrogate_dx_torch_nostuck

# Import your patched trainer
from training.glc_pinn_trainer import SteadyStatePINNTrainer


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) Build vanilla net: u -> y
    net = PINN(hidden_units=[64, 64, 64], n_u=2, n_y=3)

    # 2) Trainer
    trainer = SteadyStatePINNTrainer(
        net=net,
        surrogate_function=glc_surrogate_dx_torch_nostuck,
    )

    results = trainer.run()
    print("Done.")
    print("Best val:", results["best_val_loss"])
    print("Train at best val:", results["train_loss_at_best_val"])


if __name__ == "__main__":
    main()