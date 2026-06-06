import numpy as np
import torch
import wandb
from contextlib import nullcontext
from training.base_trainer import Trainer
from networks.networks import AlgNN
from training.dataset_generator import *
from pathlib import Path


class AlgTrainer(Trainer):
    """
    This Trainer is for the "supporting feedforward neural network" (AlgNN).
    It trains the network on a standard supervised regression task to
    map (y_states, u_controls) -> (z_algebraic_vars).
    """

    def __init__(self,
                 net: AlgNN,
                 well_list: list,
                 well_name: str,
                 N_train: int,
                 N_val: int,
                 adam_epochs: int,
                 lbfgs_epochs: int,  # Total L-BFGS iterations
                 # --- Normalization bounds from friend's 'znet_well_1_single' ---
                 # --- Other trainer params ---
                 u_min: list,
                 u_max: list,
                 lr: float = 1e-3,
                 wandb_project="PINC-GasLift-AlgNN",
                 mixed_precision=True,
                 device=None,
                 wandb_group=None,
                 random_seed=42
                 ):
        self.well_name=well_name
        self.well_list=well_list


        this_file = Path(__file__).resolve()
        project_root = this_file.parent.parent
        self.well_models_dir = project_root / "well_models"
        self.well_folder = self.well_models_dir / str(self.well_name)

        self.training_path = self.well_folder / "sweep_results_figures_training.pkl"
        self.validation_path = self.well_folder / "sweep_results_validation.pkl"

        self.BSW = well_list["BSW"]
        self.GOR = well_list["GOR"]
        self.PI = well_list["PI"]
        self.K_gs = well_list["K_gs"]
        self.K_inj = well_list["K_inj"]
        self.K_pr = well_list["K_pr"]
        self.y_guess_sur=np.array(well_list["y_guess_sur"], dtype=float)
        self.y_min_train = np.array(well_list["y_min"])
        self.y_max_train = np.array(well_list["y_max"])

        self.N_train = N_train
        self.N_val = N_val

        # Store domain bounds for data generation
        self.u_min_train = np.array(u_min)
        self.u_max_train = np.array(u_max)

        self.u1_min,self.u1_max = self.u_min_train[0], self.u_max_train[0]
        self.u2_min,self.u2_max = self.u_min_train[1], self.u_max_train[1]


        self.K1 = adam_epochs
        self.K2 = lbfgs_epochs

        self.history = {
            "epoch": [],

            "train_mse_total": [],
            "train_rmse_total": [],
            "train_rmse_z0": [],
            "train_rmse_z1": [],
            "train_rmse_z2": [],
            "train_rmse_z3": [],

            "val_mse_total": [],
            "val_rmse_total": [],
            "val_rmse_z0": [],
            "val_rmse_z1": [],
            "val_rmse_z2": [],
            "val_rmse_z3": [],
        }

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
            'z_min': net.z_min.cpu().tolist(),  # Get Z bounds from the net
            'z_max': net.z_max.cpu().tolist(),
        })

    @torch.no_grad()
    def _prediction_metrics(self, y_in, u_in, z_target):
        """
        Compute prediction metrics in physical units of z.

        z may contain:
            [P_bh_bar, P_tb_b_bar, w_G_inj, w_res]
        """

        self.net.eval()

        with self.autocast_if_mp():
            z_pred = self.net(y_in, u_in)

        error = z_pred - z_target

        mse_total = torch.mean(error ** 2)
        rmse_total = torch.sqrt(mse_total)

        mse_components = torch.mean(error ** 2, dim=0)
        rmse_components = torch.sqrt(mse_components)

        return {
            "mse_total": mse_total,
            "rmse_total": rmse_total,
            "mse_components": mse_components,
            "rmse_components": rmse_components,
            "z_pred": z_pred,
            "error": error,
        }

    def prepare_data(self):
        """
        Prepare AlgNN data from saved ground-truth sweep pickles.

        Dataset:
            input  -> (y, u)
            target -> z
        """

        self.l.info("Preparing AlgNN dataset from saved sweep pickles...")

        self.training_path = self.well_folder / "sweep_results_figures_training.pkl"
        self.validation_path = self.well_folder / "sweep_results_validation.pkl"

        if not self.training_path.exists():
            raise FileNotFoundError(f"Training sweep not found: {self.training_path}")

        if not self.validation_path.exists():
            raise FileNotFoundError(f"Validation sweep not found: {self.validation_path}")

        self.l.info(f"Loading training sweep from: {self.training_path}")
        with open(self.training_path, "rb") as f:
            results_train = pickle.load(f)

        self.l.info(f"Loading validation sweep from: {self.validation_path}")
        with open(self.validation_path, "rb") as f:
            results_val = pickle.load(f)

        batch_train = flatten_sweep_results_to_batch_full(
            results_train,
            only_success=True,
            y_names=["m_G_an", "m_G_t", "m_o_t"],
            z_names=["P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res"],
        )
        batch_val = flatten_sweep_results_to_batch_full(
            results_val,
            only_success=True,
            y_names=["m_G_an", "m_G_t", "m_o_t"],
            z_names=["P_bh_bar", "P_tb_b_bar", "w_G_inj", "w_res"],
        )

        y_train = batch_train["y_t"].to(self.device)
        u_train = batch_train["u_t"].to(self.device)
        z_train = batch_train["z_t"].to(self.device)

        y_val = batch_val["y_t"].to(self.device)
        u_val = batch_val["u_t"].to(self.device)
        z_val = batch_val["z_t"].to(self.device)

        self.train_data = (y_train, u_train, z_train)
        self.val_data = (y_val, u_val, z_val)

        self.state_names = batch_train["y_names"]
        self.target_names = batch_train["z_names"]

        self.l.info(
            f"Data preparation complete.\n"
            f"State inputs:   {self.state_names}\n"
            f"Control inputs: ['u1', 'u2']\n"
            f"Target outputs: {self.target_names}\n"
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
            with torch.no_grad():
                train_metrics = self._prediction_metrics(y_in, u_in, z_target)
                train_rmse_components = train_metrics["rmse_components"]

            losses = {
                "total": float(self._current_loss),

                "mse_total": float(train_metrics["mse_total"].item()),
                "rmse_total": float(train_metrics["rmse_total"].item()),

                "rmse_z0": float(train_rmse_components[0].item()),
                "rmse_z1": float(train_rmse_components[1].item()),
                "rmse_z2": float(train_rmse_components[2].item()),
                "rmse_z3": float(train_rmse_components[3].item()),
            }
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
            with torch.no_grad():
                train_metrics = self._prediction_metrics(y_in, u_in, z_target)
                train_rmse_components = train_metrics["rmse_components"]
            # Step the scheduler (if one exists)
            if hasattr(self, '_scheduler') and self._scheduler is not None:
                self._scheduler.step()

            losses = {
                "total": float(loss.item()),

                "mse_total": float(train_metrics["mse_total"].item()),
                "rmse_total": float(train_metrics["rmse_total"].item()),

                "rmse_z0": float(train_rmse_components[0].item()),
                "rmse_z1": float(train_rmse_components[1].item()),
                "rmse_z2": float(train_rmse_components[2].item()),
                "rmse_z3": float(train_rmse_components[3].item()),
            }

        return losses

    @torch.no_grad()
    def validation_pass(self):
        """Validation metrics in physical units."""

        self.net.eval()

        y_in_val, u_in_val, z_target_val = self.val_data

        metrics = self._prediction_metrics(
            y_in_val,
            u_in_val,
            z_target_val,
        )

        rmse_components = metrics["rmse_components"]

        return {
            "total": float(metrics["mse_total"].item()),

            "mse_total": float(metrics["mse_total"].item()),
            "rmse_total": float(metrics["rmse_total"].item()),

            "rmse_z0": float(rmse_components[0].item()),
            "rmse_z1": float(rmse_components[1].item()),
            "rmse_z2": float(rmse_components[2].item()),
            "rmse_z3": float(rmse_components[3].item()),
        }

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        self.l.info(
            f"Epoch {self._e:6d} | "
            f"train RMSE={train_losses['rmse_total']:.4e} | "
            f"val RMSE={val_losses['rmse_total']:.4e} | "
            f"val comps="
            f"[{val_losses['rmse_z0']:.3e}, "
            f"{val_losses['rmse_z1']:.3e}, "
            f"{val_losses['rmse_z2']:.3e}, "
            f"{val_losses['rmse_z3']:.3e}]"
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
            self._record_history(data_to_log)

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
            self._record_history(data_to_log)

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
        history_path = self.well_folder / "algnn_training_history.pkl"

        with open(history_path, "wb") as f:
            pickle.dump(self.history, f)

        self.l.info(f"Saved AlgNN training history at: {history_path}")
        return {
            "history": self.history,
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
            "history_path": str(history_path),
        }

    def _record_history(self, data_to_log: dict):
        """
        Store epoch-wise AlgNN metrics for later plotting.
        """

        self.history["epoch"].append(int(self._e))

        # Training metrics
        self.history["train_mse_total"].append(
            float(data_to_log.get("train/mse_total", np.nan))
        )
        self.history["train_rmse_total"].append(
            float(data_to_log.get("train/rmse_total", np.nan))
        )
        self.history["train_rmse_z0"].append(
            float(data_to_log.get("train/rmse_z0", np.nan))
        )
        self.history["train_rmse_z1"].append(
            float(data_to_log.get("train/rmse_z1", np.nan))
        )
        self.history["train_rmse_z2"].append(
            float(data_to_log.get("train/rmse_z2", np.nan))
        )
        self.history["train_rmse_z3"].append(
            float(data_to_log.get("train/rmse_z3", np.nan))
        )

        # Validation metrics
        self.history["val_mse_total"].append(
            float(data_to_log.get("val/mse_total", np.nan))
        )
        self.history["val_rmse_total"].append(
            float(data_to_log.get("val/rmse_total", np.nan))
        )
        self.history["val_rmse_z0"].append(
            float(data_to_log.get("val/rmse_z0", np.nan))
        )
        self.history["val_rmse_z1"].append(
            float(data_to_log.get("val/rmse_z1", np.nan))
        )
        self.history["val_rmse_z2"].append(
            float(data_to_log.get("val/rmse_z2", np.nan))
        )
        self.history["val_rmse_z3"].append(
            float(data_to_log.get("val/rmse_z3", np.nan))
        )


