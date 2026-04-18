import torch
import wandb
from contextlib import nullcontext

from training.base_trainer import Trainer
from networks.networks import AlgNN
from training.dataset_generator import *

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
                 y_min_train: list[float] = [3032.55, 220.05, 6341.30],
                 y_max_train: list[float] = [4796.20, 1094.60, 11990.90],
                 u_min_train: list = [0.05, 0.10],
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
        """
        Prepare AlgNN data using ground-truth sweeps.

        Dataset:
            input  -> (y,u)
            target -> (w_o_out, P_bh_bar, P_tb_b_bar)

        States y are the first 3 entries in Z_NAMES.
        Targets are selected by name.
        """

        self.l.info("Preparing AlgNN dataset from ground-truth sweeps...")

        # ------------------------------------------------
        # 1. Build sweep grids
        # ------------------------------------------------
        Nu1_train = int(np.sqrt(self.N_train))
        Nu2_train = int(np.sqrt(self.N_train))

        Nu1_val = int(np.sqrt(self.N_val))
        Nu2_val = int(np.sqrt(self.N_val))

        u1_grid_train = np.linspace(self.u_min_train[0], self.u_max_train[0], Nu1_train)
        u2_grid_train = np.linspace(self.u_min_train[1], self.u_max_train[1], Nu2_train)

        u1_grid_val = np.linspace(self.u_min_train[0], self.u_max_train[0], Nu1_val)
        u2_grid_val = np.linspace(self.u_min_train[1], self.u_max_train[1], Nu2_val)

        self.model_sur = make_model("surrogate", BSW=0.20, GOR=0.05, PI=3.0e-6)
        self.y_guess_surr = np.array([3285.42, 300.822, 6910.91], dtype=float)
        # ------------------------------------------------
        # 2. Run sweeps
        # ------------------------------------------------
        self.l.info("Running TRAIN sweep...")
        results_train = run_sweep(
            self.model_sur,
            u1_grid=u1_grid_train,
            u2_grid=u2_grid_train,
            y_guess_init=self.y_guess_surr,
            z_guess_init=None,
        )

        self.l.info("Running VAL sweep...")
        results_val = run_sweep(
            self.model_sur,
            u1_grid=u1_grid_val,
            u2_grid=u2_grid_val,
            y_guess_init=self.y_guess_surr,
            z_guess_init=None,
        )

        # ------------------------------------------------
        # 3. Flatten sweep results
        # ------------------------------------------------
        batch_train = flatten_sweep_results_to_batch(results_train, only_success=True)
        batch_val = flatten_sweep_results_to_batch(results_val, only_success=True)

        Z_NAMES = list(batch_train["Z_NAMES"])

        # Inputs
        y_train = batch_train["y_t"].to(self.device)
        u_train = batch_train["u_t"].to(self.device)

        y_val = batch_val["y_t"].to(self.device)
        u_val = batch_val["u_t"].to(self.device)

        # ------------------------------------------------
        # 4. Extract targets by name
        # ------------------------------------------------
        target_names = ["w_o_out", "P_bh_bar", "P_tb_b_bar"]

        def extract_targets(results, mask_np):
            z_cols = []
            for name in target_names:
                arr = np.asarray(results["OUT"][name], dtype=float)
                z_cols.append(arr.reshape(-1))
            z_flat = np.stack(z_cols, axis=1)
            z_np = z_flat[mask_np]
            return torch.tensor(z_np, dtype=torch.float32)

        z_train = extract_targets(results_train, batch_train["mask_np"]).to(self.device)
        z_val = extract_targets(results_val, batch_val["mask_np"]).to(self.device)

        # ------------------------------------------------
        # 5. Store datasets
        # ------------------------------------------------
        self.train_data = (y_train, u_train, z_train)
        self.val_data = (y_val, u_val, z_val)

        self.l.info(
            f"Data preparation complete.\n"
            f"Targets: {target_names}\n"
            f"Train shapes: y={tuple(y_train.shape)}, u={tuple(u_train.shape)}, z={tuple(z_train.shape)}\n"
            f"Val shapes:   y={tuple(y_val.shape)}, u={tuple(u_val.shape)}, z={tuple(z_val.shape)}"
        )

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
        """Simple validation: total MSE + per-output MSE/RMSE/MAE."""
        self.net.eval()
        (y_in_val, u_in_val, z_target_val) = self.val_data

        eps = 1e-8
        with torch.no_grad():
            with self.autocast_if_mp():
                z_pred_val = self.net(y_in_val, u_in_val)

            err = z_pred_val - z_target_val  # (N,3)

            # total
            total_mse = torch.mean(err ** 2).item()

            # per-output
            mse = torch.mean(err ** 2, dim=0)  # (3,)
            rmse = torch.sqrt(mse + eps)  # (3,)
            mae = torch.mean(torch.abs(err), dim=0)  # (3,)

        # Target order from prepare_data(): [w_o_out, P_bh_bar, P_tb_b_bar]
        return {
            "total": total_mse,

            "mse_w_o_out": mse[0].item(),
            "mse_P_bh": mse[1].item(),
            "mse_P_tb_b": mse[2].item(),

            "rmse_w_o_out": rmse[0].item(),
            "rmse_P_bh": rmse[1].item(),
            "rmse_P_tb_b": rmse[2].item(),

            "mae_w_o_out": mae[0].item(),
            "mae_P_bh": mae[1].item(),
            "mae_P_tb_b": mae[2].item(),
        }

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        """Implements the abstract logging method."""
        train_loss = train_losses['total']
        val_loss = val_losses['total']
        val_pbh_loss = val_losses.get('mse_w_o_out', 0.0)  # .get for safety

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

        # --- STAGE 1: Adam training ---
        self.l.info(f"--- Starting Stage 1: Adam training for {self.K1} epochs ---")
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

        # --- STAGE 2: L-BFGS training ---
        self.l.info(f"--- Starting Stage 2: L-BFGS training ({self.K2} iterations) ---")

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

        self.l.info('training finished!')
        return {
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
        }