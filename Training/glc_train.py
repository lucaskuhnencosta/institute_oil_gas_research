import torch
import wandb
import numpy as np
from Networks.networks import PINN
from contextlib import nullcontext
from Surrogate_ODE_Model.glc_surrogate_torch import glc_surrogate_dx_torch
from Training.base_trainer import Trainer

class SteadyStatePINNTrainer(Trainer):
    """
    This Trainer is a direct Pytorch port of the Tensorflow training
    script for the Gas-Lift WEll PINC model
    """
    def __init__(
        self,
        net: torch.nn.Module,
        surrogate_function,
        # control domain
        u_min=(0.05, 0.20),
        u_max=(1.0, 1.0),
        # collocation & validation sizes
        N_col: int = 20000,
        N_val: int = 2000,
        # optional component scaling for dx MSE
        mse_f_scale_factors=(1.0e3, 1.0e3, 1.0e2),
        u_probe=(0.6, 0.6),
        print_every: int = 1,
        # --- base Trainer kwargs below ---
        epochs: int = 3000,
        lr: float = 5e-3,
        optimizer: str = "Adam",
        optimizer_params: dict | None = None,
        loss_func: str = "MSELoss",
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        mixed_precision: bool = True,
        device=None,
        wandb_project=None,
        wandb_group=None,
        logger=None,
        checkpoint_every: int = 50,
        random_seed: int = 42,
        # optional: switch to LBFGS at epoch K1 (base Trainer.run supports this)
        K1: int | None = 2000,
        lbfgs_params: dict | None = None,
    ):
        super().__init__(
            net=net,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss_func=loss_func,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            mixed_precision=mixed_precision,
            device=device,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            logger=logger,
            checkpoint_every=checkpoint_every,
            random_seed=random_seed,
        )
        self.u_probe = np.array(u_probe, dtype=np.float32)
        self.print_every = int(print_every)
        self._u_probe_t: torch.Tensor | None = None

        # Physics function: dx = f(y, u)
        self.surrogate_function = surrogate_function

        # Domain + dataset sizes
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        self.N_col = int(N_col)
        self.N_val = int(N_val)

        self.N_u=2

        # Scale per residual component (dx1, dx2, dx3)
        self.mse_f_scale_factors = np.array(mse_f_scale_factors, dtype=np.float32)

        # Optional LBFGS switching handled by base Trainer.run()
        self.K1 = K1
        self._lbfgs_params = lbfgs_params or {}

        # Add custom info to wandb config (base Trainer will initialize wandb)
        self._add_to_wandb_config({
            "trainer": type(self).__name__,
            "u_min": self.u_min.tolist(),
            "u_max": self.u_max.tolist(),
            "N_col": self.N_col,
            "N_val": self.N_val,
            "mse_f_scale_factors": self.mse_f_scale_factors.tolist(),
            "K1": self.K1,
            "lbfgs_params": self._lbfgs_params,
        })

        # Will be created in prepare_data()
        self.u_col: torch.Tensor | None = None
        self.u_val: torch.Tensor | None = None

        # Cached tensor on device
        self._mse_f_scale_t: torch.Tensor | None = None

    def prepare_data(self):
        self.l.info("Preparing PINN data for Gas-Lift Well...")
        self.l.info(f"Generating {self.N_col} initial condition points...")

        rng = np.random.default_rng(self.random_seed)

        u_col_np = rng.uniform(self.u_min, self.u_max, size=(self.N_col, self.N_u)).astype(np.float32)
        u_val_np = rng.uniform(self.u_min, self.u_max, size=(self.N_val, self.N_u)).astype(np.float32)

        self.u_col = torch.tensor(u_col_np, dtype=torch.float32, device=self.device)
        self.u_val = torch.tensor(u_val_np, dtype=torch.float32, device=self.device)

        self._mse_f_scale_t = torch.tensor(
            self.mse_f_scale_factors, dtype=torch.float32, device=self.device
        )

        self._u_probe_t = torch.tensor(self.u_probe[None, :], dtype=torch.float32, device=self.device)

        self.l.info(
            f"Prepared data: u_col={tuple(self.u_col.shape)}, u_val={tuple(self.u_val.shape)}, "
            f"u_probe={tuple(self.u_probe.tolist())}"
        )

        self.print_shapes()

    def print_shapes(self):
        """
        Logs the shapes of all prepared data tensors for quick debugging.
        """
        self.l.info("--- Data Tensor Shapes ---")
        u_col = self.u_col
        u_val = self.u_val


        # --- Collocation (Physics) Data ---
        self.l.info(f"Collocation (Physics) Data (N_col = {u_col.shape[0]}):")
        self.l.info(f"  u_col (input):     {u_col.shape}")

        # --- Validation Data ---
        self.l.info(f"Validation Data (N_val = {u_val.shape[0]}):")
        self.l.info(f"  u_val (input):     {u_val.shape}")

        self.l.info("--------------------------")

    def switch_to_lbfgs(self,max_iter_per_call=1000):
        """Switches the optimizer to L-BFGS for fine-tuning."""
        self.l.info(f"Epoch {self._e}: Switching optimizer from Adam to L-BFGS.")
        self.optimizer = 'LBFGS'
        self.mixed_precision = False
        self.autocast_if_mp = nullcontext
        self._scaler = None

        self.lr_scheduler = None
        self._scheduler = None

        self._optim = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=max_iter_per_call,
            line_search_fn="strong_wolfe"
        )
        self._add_to_wandb_config({
            "switched_to_lbfgs_at_epoch": self._e,
            "lbfgs_max_iter_per_call": max_iter_per_call
        })
    def get_physics_loss(self, u: torch.Tensor):
        y_hat=self.net(u)
        dx=self.surrogate_function(y_hat,u)

        # component-wise MSE
        mse_components = torch.mean(dx * dx, dim=0)  # (3,)
        scaled = self._mse_f_scale_t * mse_components
        loss_f = torch.mean(scaled)

        return loss_f, mse_components

    def train_pass(self):
        self.net.train()

        if self.optimizer == 'LBFGS':
            def closure():
                self._optim.zero_grad(set_to_none=True)
                with self.autocast_if_mp():
                    loss_f, mse_f_comps = self.get_physics_loss(self.u_col)
                    # TODO (placeholder): data loss block
                    # loss_data = ...
                    # loss = loss_f + lambda_data * loss_data
                    loss=loss_f
                loss.backward()

                self._current_losses = {
                    'total': loss.item(),
                    "physics": loss_f.item(),
                    "physics_comp_0": mse_f_comps[0].item(),
                    "physics_comp_1": mse_f_comps[1].item(),
                    "physics_comp_2": mse_f_comps[2].item(),
                }

                return loss
            self._optim.step(closure)
            return self._current_losses

        if torch.is_grad_enabled():
            self._optim.zero_grad()

        with self.autocast_if_mp():
            loss_f, mse_f_comps = self.get_physics_loss(self.u_col)

            # TODO (placeholder): data loss block
            # loss_data = ...
            # loss = loss_f + lambda_data * loss_data
            loss = loss_f

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            loss.backward()
            self._optim.step()
        if self.lr_scheduler is not None:
            self._scheduler.step()
        return {
            "total": float(loss.item()),
            "physics": float(loss_f.item()),
            "physics_comp_0": float(mse_f_comps[0].item()),
            "physics_comp_1": float(mse_f_comps[1].item()),
            "physics_comp_2": float(mse_f_comps[2].item()),
        }

    @torch.no_grad()
    def validation_pass(self):
        self.net.eval()
        with self.autocast_if_mp():
            loss_f, mse_f_comps = self.get_physics_loss(self.u_val)
            # ------------------------------------------------------------
            # TODO (placeholder): DATA/OUTPUT METRICS BLOCK
            # e.g., if you have y_star or z_star:
            #   mae = mean(|g(y_hat,u) - z_star|)
            # ------------------------------------------------------------
        return {
            "total": float(loss_f.item()),
            "val_comp_0": float(mse_f_comps[0].item()),
            "val_comp_1": float(mse_f_comps[1].item()),
            "val_comp_2": float(mse_f_comps[2].item()),
        }

    def run(self):
        if not self._is_initialized:
            self.setup_training()

        # STAGE 1 - ADAM TRAINING
        self.l.info(f"--- Starting Stage 1: Adam Training for {self.K1} epochs ---")
        while self._e<self.K1:
            data_to_log,val_score=self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log,step=self._e,commit=True)
                if self._e % self.checkpoint_every==self.checkpoint_every-1:
                    self.save_checkpoint()

            if val_score<self.best_val:
                self.best_val=val_score
                self.train_loss_at_best_val=data_to_log['train/total']
                if self._log_to_wandb:
                    self.save_model(name='model_best')

            self._e+=1

        #STAGE 2 - L-BFGS Training Block 1
        self.l.info(f"--- Starting Stage 2: L-BFGS Block 1 (Adam Epochs: {self._e}) ---")
        lbfgs_iter_per_loop = 1000
        self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_loop)
        data_to_log, val_score = self._run_epoch()
        self._e += lbfgs_iter_per_loop  # Log this as 1000 new "epochs"

        if self._log_to_wandb:
            wandb.log(data_to_log, step=self._e, commit=True)
            self.save_checkpoint()
        if val_score < self.best_val:
            self.best_val = val_score
            self.train_loss_at_best_val = data_to_log['train/total']
            if self._log_to_wandb:
                self.save_model(name='model_best')

        #STAGE 3 - L-BFGS Block 2
        self.l.info(f"--- Starting Stage 3: L-BFGS Block 2 (Adam Epochs: {self._e}) ---")
        # We re-initialize the optimizer to run a new session
        self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_loop)

        data_to_log, val_score = self._run_epoch()
        self._e += lbfgs_iter_per_loop  # Log this as 1000 more "epochs"

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

    @torch.no_grad()
    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        self.net.eval()
        y_probe = self.net(self._u_probe_t).detach().cpu().numpy().reshape(-1)

        self.l.info(
            f"Epoch {self._e:6d} | "
            f"train total={train_losses['total']:.4e} "
            f"f=[{train_losses['physics_comp_0']:.3e}, {train_losses['physics_comp_1']:.3e}, {train_losses['physics_comp_2']:.3e}] | "
            f"val total={val_losses['total']:.4e} "
            f"f=[{val_losses['val_comp_0']:.3e}, {val_losses['val_comp_1']:.3e}, {val_losses['val_comp_2']:.3e}] | "
            f"u_probe={tuple(self.u_probe.tolist())} "
            f"y_hat={y_probe}"
        )
