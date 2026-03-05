import torch
import torch.nn as nn
import numpy as np
import wandb
from contextlib import nullcontext


# --- Import from our project ---
from PINC.core.base_trainer import Trainer
from PINC.core.networks import AlgNN
from PINC.experiments.Gas_Lift.glc_check_feasibility import glc_check_feasibility
from PINC.experiments.Gas_Lift.glc_f_alg import glc_f_alg


class AlgTrainer(Trainer):
    """
    This Trainer is for the "supporting feedforward neural network" (AlgNN).
    It trains the network on a standard supervised regression task to
    map (y_states, u_controls) -> (z_algebraic_vars).
    """

    def __init__(self,
                 net: AlgNN,
                 N_train: int = 10000,
                 N_val: int = 1000,
                 adam_epochs: int = 5000,
                 lbfgs_epochs: int = 5000,  # Total L-BFGS iterations
                 lr: float = 1e-3,
                 # --- Normalization bounds from friend's 'znet_well_1_single' ---
                 y_min_train: list = [3200.0, 180.0, 7000.0],
                 y_max_train: list = [4700.0, 400.0, 11000.0],
                 u_min_train: list = [0.0, 0.0],
                 u_max_train: list = [1.0, 1.0],
                 # --- Other trainer params ---
                 mixed_precision=True,
                 device=None,
                 wandb_project="PINC-GasLift-AlgNN",
                 wandb_group=None,
                 random_seed=42
                 ):

        self.N_train = N_train
        self.N_val = N_val

        # Store domain bounds for data generation
        self.y_min_train = np.array(y_min_train)
        self.y_max_train = np.array(y_max_train)
        self.u_min_train = np.array(u_min_train)
        self.u_max_train = np.array(u_max_train)

        # This is the "ground truth" function
        self.gt_f = glc_f_alg

        self.K1 = adam_epochs
        self.K2 = lbfgs_epochs

        # Call the base trainer's __init__
        # We use the base 'run' loop (no K1/K2) and a simple scheduler
        super().__init__(net=net,
                         epochs=adam_epochs + lbfgs_epochs,
                         lr=lr,
                         optimizer='Adam',
                         loss_func='MSELoss',
                         lr_scheduler='StepLR',
                         lr_scheduler_params={'step_size': 500, 'gamma': 0.8},
                         mixed_precision=mixed_precision,
                         device=device,
                         wandb_project=wandb_project,
                         wandb_group=wandb_group,
                         random_seed=random_seed)

        # Add these specific parameters to WandB
        self._add_to_wandb_config({
            'N_train': self.N_train,
            'N_val': self.N_val,
            'y_min_train': y_min_train,
            'y_max_train': y_max_train,
            'u_min_train': u_min_train,
            'u_max_train': u_max_train,
            'z_min': net.z_min.cpu().tolist(),  # Get Z bounds from the net
            'z_max': net.z_max.cpu().tolist(),
        })

    def prepare_data(self):
        self.l.info("Preparing data for AlgNN...")

        # --- 1. Generate Training Data ---
        self.l.info(f"Generating {self.N_train} training points...")
        y_train_np = np.random.uniform(self.y_min_train, self.y_max_train, (self.N_train, 3))
        u_train_np = np.random.uniform(self.u_min_train, self.u_max_train, (self.N_train, 2))

        # --- Feasibility Check (critical) ---
        y_train_torch = torch.tensor(y_train_np, dtype=torch.float32, device=self.device)
        u_train_torch = torch.tensor(u_train_np, dtype=torch.float32, device=self.device)
        feasible_mask = glc_check_feasibility(y_train_torch, u_train_torch)

        y_train_filtered = y_train_torch[feasible_mask]
        u_train_filtered = u_train_torch[feasible_mask]

        N_train_new = len(y_train_filtered)
        self.l.info(f"  Removed {self.N_train - N_train_new} infeasible points. New N_train = {N_train_new}")

        # --- Generate Targets ---
        with torch.no_grad():
            # Call our new glc_f_alg function
            z_train_targets = self.gt_f(y_train_filtered, u_train_filtered)

        self.train_data = (y_train_filtered, u_train_filtered, z_train_targets)

        # --- 2. Generate Validation Data (repeat the process) ---
        self.l.info(f"Generating {self.N_val} validation points...")
        y_val_np = np.random.uniform(self.y_min_train, self.y_max_train, (self.N_val, 3))
        u_val_np = np.random.uniform(self.u_min_train, self.u_max_train, (self.N_val, 2))

        y_val_torch = torch.tensor(y_val_np, dtype=torch.float32, device=self.device)
        u_val_torch = torch.tensor(u_val_np, dtype=torch.float32, device=self.device)
        feasible_mask_val = glc_check_feasibility(y_val_torch, u_val_torch)

        y_val_filtered = y_val_torch[feasible_mask_val]
        u_val_filtered = u_val_torch[feasible_mask_val]

        with torch.no_grad():
            z_val_targets = self.gt_f(y_val_filtered, u_val_filtered)

        self.val_data = (y_val_filtered, u_val_filtered, z_val_targets)
        self.l.info("Data preparation complete.")

    def train_pass(self):
        """Performs a single standard supervised training step."""
        self.net.train()

        (y_in, u_in, z_target) = self.train_data

        if self.optimizer == 'LBFGS':
            # --- L-BFGS PATH ---
            def closure():
                self._optim.zero_grad()
                z_pred = self.net(y_in, u_in)
                loss = self._loss_func(z_pred, z_target)
                loss.backward()
                self._current_loss = loss.item()  # Store for logging
                return loss

            self._optim.step(closure)
            losses = {'total': self._current_loss}
        else:

            if torch.is_grad_enabled():
                self._optim.zero_grad()

            with self.autocast_if_mp():
                # --- Standard Supervised Loss ---
                z_pred = self.net(y_in, u_in)
                loss = self._loss_func(z_pred, z_target)

            # Backward pass
            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optim)
                self._scaler.update()
            else:
                loss.backward()
                self._optim.step()

            # Step the scheduler (if one exists)
            if hasattr(self, '_scheduler') and self._scheduler is not None:
                self._scheduler.step()

            losses={'total': loss.item()}

        return losses

    def validation_pass(self):
        """Performs a standard validation pass."""
        self.net.eval()

        (y_in_val, u_in_val, z_target_val) = self.val_data

        with torch.no_grad():
            with self.autocast_if_mp():
                z_pred_val = self.net(y_in_val, u_in_val)
                loss = self._loss_func(z_pred_val, z_target_val)

                # Calculate component-wise MSE for logging
                # [P_bh, w_G_in, w_G_res, w_L_res]
                loss_comps = torch.mean((z_pred_val - z_target_val) ** 2, dim=0)

        return {
            'total': loss.item(),
            'comp_P_bh': loss_comps[0].item(),
            'comp_w_G_in': loss_comps[1].item(),
            'comp_w_G_res': loss_comps[2].item(),
            'comp_w_L_res': loss_comps[3].item(),
        }

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        """Implements the abstract logging method."""
        train_loss = train_losses['total']
        val_loss = val_losses['total']
        val_pbh_loss = val_losses.get('comp_P_bh', 0.0)  # .get for safety

        self.l.info(
            f"Epoch {self._e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"(Val P_bh MSE: {val_pbh_loss:.6f})"
        )

    def switch_to_lbfgs(self, max_iter_per_call=200):
        """Switches the optimizer to L-BFGS for fine-tuning."""
        self.l.info(f"Epoch {self._e}: Switching optimizer to L-BFGS (max_iter={max_iter_per_call}).")
        self.optimizer = 'LBFGS'
        self.mixed_precision = False
        self.autocast_if_mp = nullcontext
        self._scaler = None
        self.lr_scheduler = None
        self._scheduler = None

        self._optim = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=max_iter_per_call,  # 20 is default, 200 is good for one-shot
            line_search_fn="strong_wolfe"
        )
        self._add_to_wandb_config({
            "switched_to_lbfgs_at_epoch": self._e,
            "lbfgs_max_iter_per_call": max_iter_per_call
        })

    # --- NEW: Override the run method ---
    def run(self):
        """
        Custom run-loop to override the base_trainer's and implement
        an "Adam-then-LBFGS" pattern.
        """
        if not self._is_initialized:
            self.setup_training()

        # --- STAGE 1: Adam Training ---
        self.l.info(f"--- Starting Stage 1: Adam Training for {self.K1} epochs ---")
        while self._e < self.K1:
            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)
                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.save_checkpoint()

            if val_score < self.best_val:
                self.best_val = val_score
                self.train_loss_at_best_val = data_to_log['train/total']
                if self._log_to_wandb:
                    self.save_model(name='model_best')

            self._e += 1

        # --- STAGE 2: L-BFGS Training ---
        self.l.info(f"--- Starting Stage 2: L-BFGS Training ({self.K2} iterations) ---")

        # We'll do LBFGS in blocks to save checkpoints, e.g., 5 blocks of 1000 iter
        lbfgs_blocks = max(1, self.K2 // 1000)
        iter_per_block = self.K2 // lbfgs_blocks

        for i in range(lbfgs_blocks):
            self.l.info(f"--- L-BFGS Block {i + 1}/{lbfgs_blocks} ({iter_per_block} iterations) ---")
            self.switch_to_lbfgs(max_iter_per_call=iter_per_block)

            # We manually call _run_epoch() ONCE for this entire block.
            data_to_log, val_score = self._run_epoch()
            self._e += iter_per_block  # Log this as new "epochs"

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)
                self.save_checkpoint()

            if val_score < self.best_val:
                self.best_val = val_score
                self.train_loss_at_best_val = data_to_log['train/total']
                if self._log_to_wandb:
                    self.save_model(name='model_best')

        # --- FINISH ---
        if self._log_to_wandb:
            self.l.info(f"Saving final model")
            self.save_model(name='model_last')
            wandb.finish()

        self.l.info('Training finished!')
        return {
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
        }