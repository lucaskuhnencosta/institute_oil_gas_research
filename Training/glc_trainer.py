import torch

from Networks.networks import PINN
from Surrogate_ODE_Model.glc_surrogate_torch import glc_surrogate_dx_torch

# Import your patched trainer
from Training.glc_train import SteadyStatePINNTrainer


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) Build vanilla net: u -> y
    net = PINN(hidden_units=[64, 64, 64], n_u=2, n_y=3)

    # 2) Trainer
    trainer = SteadyStatePINNTrainer(
        net=net,
        surrogate_function=glc_surrogate_dx_torch,
        u_min=(0.2, 0.2),     # pick a safe range to start
        u_max=(1.0, 1.0),
        N_col=5000,           # start smaller for debugging
        N_val=1000,
        mse_f_scale_factors=(1.0, 1.0, 1.0),

        # training schedule
        epochs=2500,
        K1=2000,              # Adam for 2000 epochs, then your LBFGS blocks run

        lr=5e-3,
        optimizer="Adam",
        mixed_precision=False,  # keep False for first debug pass
        device=device,

        # disable wandb for first run
        wandb_project=None,
        wandb_group=None,
    )

    results = trainer.run()
    print("Done.")
    print("Best val:", results["best_val_loss"])
    print("Train at best val:", results["train_loss_at_best_val"])


if __name__ == "__main__":
    main()