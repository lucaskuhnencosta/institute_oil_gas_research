import torch
import wandb
import numpy as np

from contextlib import nullcontext

from PINC.core.base_trainer import Trainer
from PINC.experiments.Gas_Lift.glc_check_feasibility import glc_check_feasibility
from PINC.core.simulation_utils import simulate_full_range

# Physics residual (safeguarded)
from PINC.experiments.Gas_Lift.glc_rto_f_simplified import glc_rto_f_simplified  # <- adjust import path
# Ground-truth dynamics for data generation (full model)
from PINC.experiments.Gas_Lift.glc_f import glc_f  # <- adjust import path


class PINCWellStaticTrainer(Trainer):
    """
    Static (steady-state) hybrid training:
      - Network: u -> y*
      - Data loss: ||y_pred - y_target||^2
      - Physics loss: ||f(y_pred, u)||^2 enforcing steady state
    """

    def __init__(self,
                 net,
                 T_data: float = 2000.0,     # long horizon to reach steady state
                 N_data: int = 5000,         # supervised samples
                 N_col: int = 20000,         # collocation samples for physics
                 N_val: int = 500,
                 adam_epochs: int = 2000,
                 lbfgs_epochs: int = 2000,
                 lr: float = 3e-3,
                 lambda_phys: float = 1.0,
                 mse_f_scale_factors: list = [1.0e3, 1.0e3, 1.0e2],
                 y_min: list = [3550.0, 180.0, 7000.0],
                 y_max: list = [4500.0, 400.0, 9000.0],
                 u_min: list = [0.0, 0.0],
                 u_max: list = [1.0, 1.0],
                 data_rk4_dt: float = 10.0,
                 data_rk4_num_steps: int = 10,
                 mixed_precision=True,
                 device=None,
                 wandb_project="PINC-GasLift-Static",
                 wandb_group=None,
                 random_seed=333333):

        self.N_y = 3
        self.N_u = 2

        self.K1 = adam_epochs
        self.K2 = lbfgs_epochs

        super().__init__(
            net=net,
            epochs=adam_epochs + lbfgs_epochs,
            lr=lr,
            optimizer='Adam',
            loss_func='MSELoss',
            lr_scheduler=None,
            mixed_precision=mixed_precision,
            device=device,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            random_seed=random_seed
        )

        # Domain bounds
        self.y_min_train = np.array(y_min)
        self.y_max_train = np.array(y_max)
        self.u_min_train = np.array(u_min)
        self.u_max_train = np.array(u_max)

        # Data generation settings
        self.T_data = T_data
        self.N_data = N_data
        self.N_col = N_col
        self.N_val = N_val
        self.data_rk4_dt = data_rk4_dt
        self.data_rk4_num_steps = data_rk4_num_steps

        # Loss weights
        self.lambda_phys = lambda_phys
        self.mse_f_scale = torch.tensor(mse_f_scale_factors, device=self.device)

        # Physics and GT models
        self.physics_f = glc_rto_f_simplified   # safeguarded residual model for training
        self.gt_f = glc_f                       # “truth” simulator for label generation

        self._add_to_wandb_config({
            "T_data": self.T_data,
            "N_data": self.N_data,
            "N_col": self.N_col,
            "N_val": self.N_val,
            "K1_adam": self.K1,
            "K2_lbfgs": self.K2,
            "lambda_phys": self.lambda_phys,
            "mse_f_scale_factors": mse_f_scale_factors,
            "y_min": y_min,
            "y_max": y_max,
            "u_min": u_min,
            "u_max": u_max
        })

    # -------------------------
    # Data prep
    # -------------------------
    def prepare_data(self):
        self.l.info("Preparing STATIC (steady-state) training data...")

        np.random.seed(self.random_seed)

        # -------------------------
        # 1) Supervised data: (u -> y*)
        # -------------------------
        self.l.info(f"Generating {self.N_data} supervised steady-state samples via simulation...")

        u_list, y_target_list = [], []

        # Choose one “standard” feasible initial condition OR random feasible ICs.
        # Here we sample y0 randomly and check feasibility.
        with torch.no_grad():
            i = 0
            while i < self.N_data:
                u_np = np.random.uniform(self.u_min_train, self.u_max_train, (1, self.N_u))
                y0_np = np.random.uniform(self.y_min_train, self.y_max_train, (1, self.N_y))

                y0 = torch.tensor(y0_np, dtype=torch.float32, device=self.device)
                u = torch.tensor(u_np, dtype=torch.float32, device=self.device)

                # Ensure start feasible
                if not glc_check_feasibility(y0, u).item():
                    continue

                # Simulate long enough to approach steady state
                t_traj, y_traj = simulate_full_range(
                    f=self.gt_f,
                    y0=y0,
                    u0=u,
                    simulation_range=self.T_data,
                    num_inner_steps=self.data_rk4_num_steps,
                    dt_step=self.data_rk4_dt,
                    device=self.device
                )

                yT = y_traj[-1:]

                # Filter: no NaNs, feasible endpoint
                if torch.isnan(yT).any():
                    continue
                if not glc_check_feasibility(yT, u).item():
                    continue

                # Optional: ensure "close to steady state" under safeguarded physics
                # (helps remove not-converged trajectories)
                res = self.physics_f(yT, u)
                if torch.isnan(res).any():
                    continue
                # loose tolerance; adjust as needed
                if torch.norm(res, p=2).item() > 1e2:
                    continue

                u_list.append(u)
                y_target_list.append(yT)
                i += 1

        self.u_data = torch.cat(u_list, dim=0)
        self.y_target = torch.cat(y_target_list, dim=0)

        # Split train/val
        idx = torch.randperm(self.u_data.shape[0], device=self.device)
        n_val = min(self.N_val, self.u_data.shape[0] // 10 if self.u_data.shape[0] >= 10 else 0)

        val_idx = idx[:n_val]
        trn_idx = idx[n_val:]

        self.u_train = self.u_data[trn_idx]
        self.y_train = self.y_target[trn_idx]

        self.u_val = self.u_data[val_idx]
        self.y_val = self.y_target[val_idx]

        self.l.info(f"Supervised set: train={self.u_train.shape[0]} val={self.u_val.shape[0]}")

        # -------------------------
        # 2) Physics collocation: u only
        # -------------------------
        self.l.info(f"Generating {self.N_col} collocation points in control space...")
        u_col_np = np.random.uniform(self.u_min_train, self.u_max_train, (self.N_col, self.N_u))
        self.u_col = torch.tensor(u_col_np, dtype=torch.float32, device=self.device)

        self.print_shapes()

    def print_shapes(self):
        self.l.info("--- Data Tensor Shapes (STATIC) ---")
        self.l.info(f"u_train:   {self.u_train.shape}")
        self.l.info(f"y_train:   {self.y_train.shape}")
        self.l.info(f"u_val:     {self.u_val.shape}")
        self.l.info(f"y_val:     {self.y_val.shape}")
        self.l.info(f"u_col:     {self.u_col.shape}")
        self.l.info("--------------------------")

    # -------------------------
    # Optimizer switching
    # -------------------------
    def switch_to_lbfgs(self, max_iter_per_call=1000):
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

    # -------------------------
    # Losses
    # -------------------------
    def get_data_loss(self, u, y_target):
        y_pred = self.net(u)  # net: u -> y

        y_range = (self.net.y_max - self.net.y_min)  # keep your normalization style
        loss_components = torch.mean(((y_pred - y_target) / y_range) ** 2, dim=0)
        loss = torch.mean(loss_components)
        return loss, loss_components

    def get_physics_loss(self, u_col):
        """
        Enforce steady-state: f(y(u), u) = 0
        """
        y_pred = self.net(u_col)
        f_res = self.physics_f(y_pred, u_col)  # shape (N_col, 3)

        y_range = (self.net.y_max - self.net.y_min)
        loss_components = torch.mean((f_res / y_range) ** 2, dim=0)
        scaled = self.mse_f_scale * loss_components
        loss = torch.mean(scaled)
        return loss, loss_components

    # -------------------------
    # Train / Validation
    # -------------------------
    def train_pass(self):
        self.net.train()

        u_tr = self.u_train
        y_tr = self.y_train
        u_col = self.u_col

        if self.optimizer == 'LBFGS':
            def closure():
                self._optim.zero_grad()

                loss_data, data_comps = self.get_data_loss(u_tr, y_tr)
                loss_phys, phys_comps = self.get_physics_loss(u_col)

                loss = loss_data + self.lambda_phys * loss_phys
                loss.backward()

                self._current_losses = {
                    "total": loss.item(),
                    "data": loss_data.item(),
                    "physics": loss_phys.item(),
                    "data_comp_0": data_comps[0].item(),
                    "data_comp_1": data_comps[1].item(),
                    "data_comp_2": data_comps[2].item(),
                    "phys_comp_0": phys_comps[0].item(),
                    "phys_comp_1": phys_comps[1].item(),
                    "phys_comp_2": phys_comps[2].item(),
                }
                return loss

            self._optim.step(closure)
            return self._current_losses

        else:
            if torch.is_grad_enabled():
                self._optim.zero_grad()

            with self.autocast_if_mp():
                loss_data, data_comps = self.get_data_loss(u_tr, y_tr)
                loss_phys, phys_comps = self.get_physics_loss(u_col)

                loss = loss_data + self.lambda_phys * loss_phys

            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optim)
                self._scaler.update()
            else:
                loss.backward()
                self._optim.step()

            return {
                "total": loss.item(),
                "data": loss_data.item(),
                "physics": loss_phys.item(),
                "data_comp_0": data_comps[0].item(),
                "data_comp_1": data_comps[1].item(),
                "data_comp_2": data_comps[2].item(),
                "phys_comp_0": phys_comps[0].item(),
                "phys_comp_1": phys_comps[1].item(),
                "phys_comp_2": phys_comps[2].item(),
            }

    def validation_pass(self):
        self.net.eval()
        with torch.no_grad():
            with self.autocast_if_mp():
                y_pred = self.net(self.u_val)
                y_range = (self.net.y_max - self.net.y_min)
                val_comps = torch.mean(((y_pred - self.y_val) / y_range) ** 2, dim=0)
                val_loss = torch.mean(val_comps).item()

        return {
            "total": val_loss,
            "val_comp_0": val_comps[0].item(),
            "val_comp_1": val_comps[1].item(),
            "val_comp_2": val_comps[2].item(),
        }

    # -------------------------
    # Run loop (Adam -> LBFGS)
    # -------------------------
    def run(self):
        if not self._is_initialized:
            self.setup_training()

        # Stage 1: Adam
        self.l.info(f"--- Stage 1: Adam for {self.K1} epochs ---")
        while self._e < self.K1:
            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)
                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.save_checkpoint()

            if val_score < self.best_val:
                self.best_val = val_score
                self.train_loss_at_best_val = data_to_log["train/total"]
                if self._log_to_wandb:
                    self.save_model(name="model_best")

            self._e += 1

        # Stage 2: LBFGS block(s)
        lbfgs_iter_per_loop = 1000

        self.l.info("--- Stage 2: L-BFGS Block 1 ---")
        self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_loop)
        data_to_log, val_score = self._run_epoch()
        self._e += lbfgs_iter_per_loop

        if self._log_to_wandb:
            wandb.log(data_to_log, step=self._e, commit=True)
            self.save_checkpoint()
        if val_score < self.best_val:
            self.best_val = val_score
            self.train_loss_at_best_val = data_to_log["train/total"]
            if self._log_to_wandb:
                self.save_model(name="model_best")

        self.l.info("--- Stage 3: L-BFGS Block 2 ---")
        self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_loop)
        data_to_log, val_score = self._run_epoch()
        self._e += lbfgs_iter_per_loop

        if self._log_to_wandb:
            wandb.log(data_to_log, step=self._e, commit=True)
            self.save_checkpoint()
        if val_score < self.best_val:
            self.best_val = val_score
            self.train_loss_at_best_val = data_to_log["train/total"]
            if self._log_to_wandb:
                self.save_model(name="model_best")

        if self._log_to_wandb:
            self.l.info("Saving final model")
            self.save_model(name="model_last")
            wandb.finish()

        self.l.info("Training finished!")
        return {
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
        }

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        self.l.info(
            f"Epoch {self._e} | "
            f"Train: total={train_losses['total']:.8f} "
            f"(data={train_losses['data']:.8f}, phys={train_losses['physics']:.8f}) | "
            f"Val: total={val_losses['total']:.8f}"
        )
