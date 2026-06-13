import torch
from torch.optim import lbfgs

import wandb
import numpy as np
from networks.networks import PINN
from contextlib import nullcontext
from training.base_trainer import Trainer
from application.simulation_engine import make_model
from application.simulation_engine import run_sweep
from settings import *
from pathlib import Path
import pickle

from simulators.surrogate_simulator.surrogate_model_torch import Z_DIAG_NAMES


class SteadyStatePINNTrainer(Trainer):
    def __init__(
            self,
            net: PINN,
            surrogate_function,
            well_name: str,
            well_list: dict,
            N_col: int,
            lambda_data: float,
            lambda_physics: float,
            adam_epochs_1: int,
            adam_epochs_2: int,
            lbfgs_epochs: int,
            lr: float,
            mse_f_scale_factors: list,
            wandb_project="PINC_GasLift",
            mixed_precision=True,
            device=None,
            wandb_group=None,
            random_seed=333333,
            well_models_dir=None):

        self.dx_scale_t = torch.tensor(
            [10.0, 10.0, 10.0],
            dtype=torch.float32,
        )

        self.lbfgs_lr = 0.05

        self.lr_stage2 = 5e-5

        # self.use_data_points_as_collocation= False,
        # ------------------------------------------------
        # 1. Well identification and metadata
        # ------------------------------------------------

        self.well_name = well_name
        self.well_list = well_list

        self.BSW = well_list["BSW"]
        self.GOR = well_list["GOR"]
        self.PI = well_list["PI"]
        self.K_gs = well_list["K_gs"]
        self.K_inj = well_list["K_inj"]
        self.K_pr = well_list["K_pr"]

        self.y_guess_surr=np.array(well_list["y_guess_sur"], dtype=float)

        self.y_min_train = np.array(well_list["y_min"],dtype=float)
        self.y_max_train = np.array(well_list["y_max"],dtype=float)

        # ------------------------------------------------
        # 2. Problem dimensions
        # ------------------------------------------------

        self.N_y=3
        self.N_u=2

        # ------------------------------------------------
        # 3. Training hyperparameters
        # ------------------------------------------------

        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics

        self.K1=adam_epochs_1
        self.K2=adam_epochs_2
        self.K3=lbfgs_epochs

        self.N_col = N_col

        # ------------------------------------------------
        # 4. Input domain
        # ------------------------------------------------
        # self.U1_MIN=0.10
        # self.U2_MIN=0.20
        #
        self.U1_MIN=U1_MIN
        self.U2_MIN=U2_MIN
        self.u_min_train = np.array([self.U1_MIN,self.U2_MIN], dtype=float)
        self.u_max_train = np.array([U1_MAX,U2_MAX],dtype=float)

        self.u1_min,self.u1_max = self.u_min_train[0], self.u_max_train[0]
        self.u2_min,self.u2_max = self.u_min_train[1], self.u_max_train[1]


        # ------------------------------------------------
        # 5. Paths to saved datasets
        # ------------------------------------------------

        if well_models_dir is None:
            this_file = Path(__file__).resolve()
            project_root = this_file.parent.parent
            self.well_models_dir = project_root / "well_models"
        else:
            self.well_models_dir = Path(well_models_dir)

        self.well_folder = self.well_models_dir / str(self.well_name)
        self.poly_dataset_path = self.well_folder / "poly_dataset.pkl"
        self.validation_path = self.well_folder / "sweep_results_figures_training.pkl"
        self.training_path = self.well_folder / "sweep_results.pkl"

        # ------------------------------------------------
        # 6. Base trainer initialization
        # ------------------------------------------------

        super().__init__(
            net=net,
            epochs=adam_epochs_1+adam_epochs_2+lbfgs_epochs,
            lr=lr,
            optimizer='Adam',
            loss_func='MSELoss',
            lr_scheduler=None,
            mixed_precision=mixed_precision,
            device=device,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            random_seed=random_seed,
        )

        # ------------------------------------------------
        # 7. Physics residual settings
        # ------------------------------------------------
        self.physics_f = surrogate_function
        self.mse_f_scale = torch.tensor(
            mse_f_scale_factors,
            dtype=torch.float32,
            device=self.device,
        )

        # ------------------------------------------------
        # 8. Explicit history for plotting later
        # ------------------------------------------------
        self.history = {
            "epoch": [],

            # Training objective terms
            "train_total": [],
            "train_data": [],
            "train_physics": [],

            # Training RMSE in physical units
            "train_rmse_total": [],
            "train_rmse_y0": [],
            "train_rmse_y1": [],
            "train_rmse_y2": [],

            # Validation RMSE in physical units
            "val_rmse_total": [],
            "val_rmse_y0": [],
            "val_rmse_y1": [],
            "val_rmse_y2": [],
        }

        # ------------------------------------------------
        # 9. W&B metadata
        # ------------------------------------------------
        self._add_to_wandb_config({
            "well_name": self.well_name,
            "N_col": self.N_col,
            "K1_adam": self.K1,
            "K2_adam": self.K2,
            "K3_lbfgs": self.K3,
            "lambda_data": self.lambda_data,
            "lambda_physics": self.lambda_physics,
            "u_min": list(self.u_min_train),
            "u_max": list(self.u_max_train),
            "mse_f_scale_factors": mse_f_scale_factors,
        })

    def _flatten_sweep_to_u_y(self, sweep_results, output_names):
        u1_grid = np.asarray(sweep_results["u1_grid"], dtype=float)
        u2_grid = np.asarray(sweep_results["u2_grid"], dtype=float)

        u_list = []
        y_list = []

        for i, u1 in enumerate(u1_grid):
            for j, u2 in enumerate(u2_grid):
                y_row = [
                    sweep_results["OUT"][name][i, j]
                    for name in output_names
                ]

                y_row = np.asarray(y_row, dtype=float)

                if np.all(np.isfinite(y_row)):
                    u_list.append([u1, u2])
                    y_list.append(y_row)

        u_np = np.asarray(u_list, dtype=np.float32)
        y_np = np.asarray(y_list, dtype=np.float32)

        if u_np.shape[1] != self.N_u:
            raise ValueError(
                f"Expected u_np to have {self.N_u} columns, got {u_np.shape}."
            )

        if y_np.shape[1] != self.N_y:
            raise ValueError(
                f"Expected y_np to have {self.N_y} columns, got {y_np.shape}."
            )

        return u_np, y_np

    def prepare_data(self):
        self.l.info(f"Preparing PINN data for Gas-Lift Well {self.well_name}...")
        # ------------------------------------------------
        # 1. Output scaling tensors
        # ------------------------------------------------
        self.y_min_t = torch.tensor(self.y_min_train,
                                    dtype=torch.float32,
                                    device=self.device)
        self.y_max_t = torch.tensor(self.y_max_train,
                                    dtype=torch.float32,
                                    device=self.device)
        self.y_range_t = (self.y_max_t - self.y_min_t).clamp_min(1e-6)

        # ------------------------------------------------
        # 2. Load the supervised data training
        # ------------------------------------------------

        self.l.info(f"Loading the 17-point PINN supervised dataset:")
        self.l.info(str(self.poly_dataset_path))

        if not self.poly_dataset_path.exists():
            raise FileNotFoundError(
                f"Could not find poly_dataset.pkl for well {self.well_name}: "
                f"{self.poly_dataset_path}"
            )

        with open(self.poly_dataset_path, "rb") as f:
            poly_data = pickle.load(f)

        output_names = ["m_G_an", "m_G_t", "m_o_t"]

        u_train_np = np.asarray(poly_data["U"], dtype=np.float32)
        y_train_np = np.asarray(poly_data["Y_pinn"], dtype=np.float32)

        self.train_output_names = list(output_names)

        self.l.info(
            f"Loaded 17-point PINN dataset: "
            f"u_train={u_train_np.shape}, y_train={y_train_np.shape}"
        )

        if u_train_np.shape != (17, self.N_u):
            raise ValueError(
                f"Expected u_train_np shape (17, {self.N_u}), got {u_train_np.shape}."
            )

        if y_train_np.shape != (17, self.N_y):
            raise ValueError(
                f"Expected y_train_np shape (17, {self.N_y}), got {y_train_np.shape}."
            )

        self.u_data_train = torch.tensor(
            u_train_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_data_train = torch.tensor(
            y_train_np,
            dtype=torch.float32,
            device=self.device,
        )

        # self.l.info(f"Loading the supervised data training dataset:")
        # self.l.info(str(self.training_path))
        #
        # if not self.training_path.exists():
        #     raise FileNotFoundError(
        #         f"Could not find sweep_results.pkl for well {self.well_name}: "
        #         f"{self.training_path}"
        #     )
        #
        # with open(self.training_path, "rb") as f:
        #     sweep_results = pickle.load(f)
        #
        # output_names = ["m_G_an", "m_G_t", "m_o_t"]
        #
        # u_full_np, y_full_np = self._flatten_sweep_to_u_y(
        #     sweep_results,
        #     output_names,
        # )
        #
        # self.train_output_names = list(output_names)
        #
        # self.l.info(
        #     f"Loaded full sweep dataset: "
        #     f"u_full={u_full_np.shape}, y_full={y_full_np.shape}"
        # )
        #
        # self.u_data_train = torch.tensor(
        #     u_full_np,
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        #
        # self.y_data_train = torch.tensor(
        #     y_full_np,
        #     dtype=torch.float32,
        #     device=self.device,
        # )

        self.y_data_train_norm = (self.y_data_train - self.y_min_t)/self.y_range_t

        self.l.info("========== 17-POINT DATASET DEBUG ==========")
        self.l.info(f"u_data_train shape: {tuple(self.u_data_train.shape)}")
        self.l.info(f"y_data_train shape: {tuple(self.y_data_train.shape)}")
        self.l.info(f"u_data_train min: {torch.min(self.u_data_train, dim=0).values}")
        self.l.info(f"u_data_train max: {torch.max(self.u_data_train, dim=0).values}")
        self.l.info(f"y_data_train min: {torch.min(self.y_data_train, dim=0).values}")
        self.l.info(f"y_data_train max: {torch.max(self.y_data_train, dim=0).values}")
        self.l.info(f"First 5 u_data_train:\n{self.u_data_train[:5]}")
        self.l.info(f"First 5 y_data_train:\n{self.y_data_train[:5]}")
        self.l.info("============================================")

        # # -----------------------------
        # # 3) Physics collocation points
        # # -----------------------------
        #
        # self.l.info(f"Generating {self.N_col} random physics collocation points.")
        # np.random.seed(333333)
        #
        # u_col_extra_np = np.random.uniform(self.u_min_train,
        #                              self.u_max_train,
        #                              size=(self.N_col, self.N_u)
        #                              ).astype(np.float32)
        #
        # self.u_col = torch.tensor(u_col_extra_np,
        #                           dtype=torch.float32,
        #                           device=self.device)

        # -----------------------------
        # 3) Physics collocation points
        # -----------------------------

        self.l.info(f"Generating {self.N_col} LHS physics collocation points.")

        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=self.N_u, seed=333333)
        u_unit = sampler.random(n=self.N_col).astype(np.float32)

        u_col_extra_np = qmc.scale(
            u_unit,
            self.u_min_train,
            self.u_max_train
        ).astype(np.float32)

        self.u_col = torch.tensor(
            u_col_extra_np,
            dtype=torch.float32,
            device=self.device
        )

        # ------------------------------------------------
        # 4. Load validation sweep dataset
        # ------------------------------------------------
        self.l.info(f"Loading the validation dataset:")
        self.l.info(str(self.validation_path))

        if not self.validation_path.exists():
            raise FileNotFoundError(
                f"Could not find sweep_results_validation.pkl for well {self.well_name}: "
                f"{self.validation_path}"
            )

        with open(self.validation_path, "rb") as f:
            sweep_results_validation = pickle.load(f)

        u_val_np, y_val_np = self._flatten_sweep_to_u_y(
            sweep_results_validation,
            output_names,
        )

        self.l.info(
            f"Loaded validation sweep dataset: "
            f"u_val={u_val_np.shape}, y_val={y_val_np.shape}"
        )

        self.u_data_val = torch.tensor(
            u_val_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_data_val = torch.tensor(
            y_val_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_data_val_norm = (self.y_data_val - self.y_min_t) / self.y_range_t

        # ------------------------------------------------
        # 5. Store metadata for later plotting/debugging
        # ------------------------------------------------

        self.l.info(
            f"Prepared data for well {self.well_name}: "
            f"u_data_train={tuple(self.u_data_train.shape)}, "
            f"y_data_train={tuple(self.y_data_train.shape)}, "
            f"u_col_data={tuple(self.u_col.shape)}, "
            f"u_data_val={tuple(self.u_data_val.shape)}, "
            f"y_data_val={tuple(self.y_data_val.shape)}, "
        )
        self.print_shapes()
        self.debug_physics_at_data_points()

    def print_shapes(self):
        self.l.info(f"Size of y_data_train: {tuple(self.y_data_train.shape)}")
        self.l.info(f"Size of u_data_train: {tuple(self.u_data_train.shape)}")

        self.l.info(f"Size of y_data_val: {tuple(self.y_data_val.shape)}")
        self.l.info(f"Size of u_data_val: {tuple(self.u_data_val.shape)}")

        self.l.info(f"Size of collocation points: {tuple(self.u_col.shape)}")

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
            lr=self.lbfgs_lr,
            max_iter=max_iter_per_call,
            line_search_fn="strong_wolfe"
        )

        self._add_to_wandb_config({
            "switched_to_lbfgs_at_epoch": self._e,
            "lbfgs_max_iter_per_call": max_iter_per_call,
            "lbfgs_lr": self.lbfgs_lr,
        })

    def get_physics_loss(self, u: torch.Tensor):
        # ------------------------------------------------
        # 1. Predict steady-state variables/outputs
        # ------------------------------------------------

        y_hat=self.net(u)

        # ------------------------------------------------
        # 2. Evaluate physics residual
        # ------------------------------------------------

        physics_out=self.physics_f(y_hat,
                          u,
                          BSW=self.BSW,
                          GOR=self.GOR,
                          PI=self.PI,
                          K_gs=self.K_gs,
                          K_inj=self.K_inj,
                          K_pr=self.K_pr,
                          return_z=True  )

        if isinstance(physics_out, tuple):
            dx = physics_out[0]
            z = physics_out[1]
        else:
            dx = physics_out
            z = None
        # ------------------------------------------------
        # 3. Scale residual components
        # ------------------------------------------------

        y_range = (self.y_max_t - self.y_min_t)
        # dx_scaled = dx/y_range
        dx_scaled=dx/self.dx_scale_t




        # ------------------------------------------------
        # 4. Component-wise residual MSE
        # ------------------------------------------------
        mse_components = torch.mean(dx_scaled ** 2, dim=0)


        # ------------------------------------------------
        # 5. Weighted scalar physics loss
        # ------------------------------------------------
        loss_f = torch.mean(self.mse_f_scale * mse_components)


        return loss_f, mse_components, y_hat, dx

    def _data_loss(self):
        y_pred = self.net(self.u_data_train)
        y_pred_norm = (y_pred - self.y_min_t) / self.y_range_t

        err = y_pred_norm - self.y_data_train_norm

        weights = torch.tensor(
            [1.0, 1.0, 10.0],
            dtype=torch.float32,
            device=self.device,
        )


        # loss_data = torch.mean((y_pred_norm - self.y_data_train_norm) ** 2)
        loss_data = torch.mean(weights * err ** 2)
        return  loss_data, y_pred_norm, y_pred

    def _prediction_metrics(self,u,y_true):

        y_pred=self.net(u)
        error = y_pred - y_true

        mse_total = torch.mean(error ** 2)
        rmse_total = torch.sqrt(mse_total)

        mse_components = torch.mean(error ** 2, dim=0)
        rmse_components = torch.sqrt(mse_components)

        return {
            "mse_total": mse_total,
            "rmse_total": rmse_total,
            "rmse_components": rmse_components,
            "y_pred": y_pred,
            "error": error,
        }

    def _validation_metrics(self):
        return self._prediction_metrics(
            self.u_data_val,
            self.y_data_val,
        )

    def _training_prediction_metrics(self):
        return self._prediction_metrics(
            self.u_data_train,
            self.y_data_train,
        )

    def train_pass(self):
        self.net.train()

        # ------------------------------------------------
        # Helper: decide training stage
        # ------------------------------------------------
        if self._e < self.K1:
            stage = 1
        elif self._e < self.K1 + self.K2:
            stage = 2

        # ------------------------------------------------
        # 1. L-BFGS stage
        # ------------------------------------------------

        if self.optimizer == 'LBFGS':

            def closure():
                self._optim.zero_grad(set_to_none=True)
                with self.autocast_if_mp():
                    loss_data_n, y_pred_data_norm, y_pred_data = self._data_loss()
                    loss_f, mse_f, _, _ = self.get_physics_loss(self.u_col)
                    loss = self.lambda_data * loss_data_n + self.lambda_physics*loss_f
                    train_metrics = self._training_prediction_metrics()
                    train_rmse_components = train_metrics["rmse_components"]
                loss.backward()
                self._current_losses = {
                    'total': float(loss.item()),
                    "physics": float(loss_f.item()),
                    "data": float(loss_data_n.item()),
                    "rmse_total": float(train_metrics["rmse_total"].item()),
                    "rmse_y0": float(train_rmse_components[0].item()),
                    "rmse_y1": float(train_rmse_components[1].item()),
                    "rmse_y2": float(train_rmse_components[2].item()),
                }
                return loss
            self._optim.step(closure)
            return self._current_losses

        # ------------------------------------------------
        # 2. Adam stage
        # ------------------------------------------------
        if torch.is_grad_enabled():
            self._optim.zero_grad(set_to_none=True)

        with self.autocast_if_mp():
            loss_data_n, y_pred_data_norm, y_pred_data = self._data_loss()
            loss_f, mse_f, _, _ = self.get_physics_loss(self.u_col)
            if stage == 1:
                lambda_f = 0.0
            elif stage == 2:
                lambda_f = self.lambda_physics
            loss = self.lambda_data*loss_data_n + lambda_f*loss_f
            train_metrics = self._training_prediction_metrics()
            train_rmse_components = train_metrics["rmse_components"]
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            # # Optional: for AMP, gradient clipping requires unscale first
            # self._scaler.unscale_(self._optim)
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self._optim.step()

        # ------------------------------------------------
        # 4. Scheduler step, if used
        # ------------------------------------------------

        if self.lr_scheduler is not None:
            self._scheduler.step()

        # ------------------------------------------------
        # 5. Return scalar losses for logging
        # ------------------------------------------------
        return {
        'total': float(loss.item()),
        "physics": float(loss_f.item()),
        "data": float(loss_data_n.item()),
        "rmse_total": float(train_metrics["rmse_total"].item()),
        "rmse_y0": float(train_rmse_components[0].item()),
        "rmse_y1": float(train_rmse_components[1].item()),
        "rmse_y2": float(train_rmse_components[2].item()),
        }

    @torch.no_grad()
    def validation_pass(self):
        self.net.eval()
        with self.autocast_if_mp():
            metrics = self._validation_metrics()
        rmse_components = metrics["rmse_components"]
        return {
            # Scalar score used by Trainer for best-val tracking
            "total": float(metrics["mse_total"].item()),

            # Raw physical-unit validation metrics
            "rmse_total": float(metrics["rmse_total"].item()),

            "rmse_y0": float(rmse_components[0].item()),
            "rmse_y1": float(rmse_components[1].item()),
            "rmse_y2": float(rmse_components[2].item()),
        }

    def _record_history(self, data_to_log: dict):
        """
        Store epoch-wise losses in self.history for later plotting.
        """
        self.history["epoch"].append(int(self._e))

        # -----------------------------
        # Training losses
        # -----------------------------
        self.history["train_total"].append(float(data_to_log.get("train/total", np.nan)))
        self.history["train_data"].append(float(data_to_log.get("train/data", np.nan)))
        self.history["train_physics"].append(float(data_to_log.get("train/physics", np.nan)))

        self.history["train_rmse_total"].append(float(data_to_log.get("train/rmse_total", np.nan)))
        self.history["train_rmse_y0"].append(float(data_to_log.get("train/rmse_y0", np.nan)))
        self.history["train_rmse_y1"].append(float(data_to_log.get("train/rmse_y1", np.nan)))
        self.history["train_rmse_y2"].append(float(data_to_log.get("train/rmse_y2", np.nan)))

        # -----------------------------
        # Validation losses: raw units
        # -----------------------------
        self.history["val_rmse_total"].append(float(data_to_log.get("val/rmse_total", np.nan)))
        self.history["val_rmse_y0"].append(float(data_to_log.get("val/rmse_y0", np.nan)))
        self.history["val_rmse_y1"].append(float(data_to_log.get("val/rmse_y1", np.nan)))
        self.history["val_rmse_y2"].append(float(data_to_log.get("val/rmse_y2", np.nan)))

    def reset_adam(self, lr):
        self.optimizer = "Adam"
        self._optim = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
        )

        self.l.info(f"Reset Adam optimizer with lr={lr:.3e}")

    def run(self):
        if not self._is_initialized:
            self.setup_training()

        total_adam_epochs = self.K1 + self.K2
        # ------------------------------------------------
        # Stage 1: Adam training
        # ------------------------------------------------
        self.l.info(f"--- Starting Stage 1: Adam training for {self.K1} epochs ---")
        while self._e<total_adam_epochs:
            if self._e == self.K1:
                self.reset_adam(lr=self.lr_stage2)
            data_to_log,val_score=self._run_epoch()
            self._record_history(data_to_log)
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

        # ------------------------------------------------
        # Stage 2: L-BFGS fine-tuning
        # ------------------------------------------------
        self.l.info(f"--- Starting Stage 3: L-BFGS fine-tuning for {self.K2} iterations ---")
        lbfgs_iter_per_call = self.K3
        self.switch_to_lbfgs(max_iter_per_call=lbfgs_iter_per_call)
        data_to_log, val_score = self._run_epoch()
        self._e +=lbfgs_iter_per_call  # Log this as 1000 new "epochs"
        self._record_history(data_to_log)
        if self._log_to_wandb:
            wandb.log(data_to_log, step=self._e, commit=True)
            self.save_checkpoint()
        if val_score < self.best_val:
            self.best_val = val_score
            self.train_loss_at_best_val = data_to_log['train/total']
            if self._log_to_wandb:
                self.save_model(name='model_best')

        # ------------------------------------------------
        # Finish
        # ------------------------------------------------
        if self._log_to_wandb:
            self.l.info(f"Saving final model")
            self.save_model(name='model_last')
            wandb.finish()
        # ------------------------------------------------
        # Save local history for later plotting
        # ------------------------------------------------
        history_path = self.well_folder / "pinn_training_history.pkl"

        with open(history_path, "wb") as f:
            pickle.dump(self.history, f)

        self.l.info(f"Saved PINN training history at: {history_path}")

        self.l.info("Training finished!")
        return {
            "history": self.history,
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
            "history_path": str(history_path),
        }

    @torch.no_grad()
    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        """
        Print compact epoch information during training.

        train_losses comes from train_pass().
        val_losses comes from validation_pass().
        """

        self.net.eval()

        self.l.info(
            f"Epoch {self._e:6d} | "
            f"train total={train_losses['total']:.4e} "
            f"(data={train_losses['data']:.3e}, "
            f"phys={train_losses['physics']:.3e}, "
            f"train rmse={train_losses['rmse_total']:.3e}, "
            f"val mse={val_losses['total']:.4e}, "
            f"val rmse={val_losses['rmse_total']:.4e}, "
            f"val rmse comps="
            f"[{val_losses['rmse_y0']:.3e}, "
            f"{val_losses['rmse_y1']:.3e}, "
            f"{val_losses['rmse_y2']:.3e}] | "
        )

    @torch.no_grad()
    def debug_physics_at_data_points(self):
        self.net.eval()

        physics_out = self.physics_f(
            self.y_data_train,
            self.u_data_train,
            BSW=self.BSW,
            GOR=self.GOR,
            PI=self.PI,
            K_gs=self.K_gs,
            K_inj=self.K_inj,
            K_pr=self.K_pr,
            return_z=True,
        )

        if isinstance(physics_out, tuple):
            dx = physics_out[0]
            z = physics_out[1]
        else:
            dx = physics_out
            z = None

        dx_scaled = dx / self.dx_scale_t
        mse_components = torch.mean(dx_scaled ** 2, dim=0)
        loss_f = torch.mean(self.mse_f_scale * mse_components)

        self.l.info("========== PHYSICS AT TRUE DATA POINTS ==========")
        self.l.info(f"u_data_train shape: {tuple(self.u_data_train.shape)}")
        self.l.info(f"y_data_train shape: {tuple(self.y_data_train.shape)}")

        self.l.info(f"dx min: {torch.min(dx, dim=0).values}")
        self.l.info(f"dx max: {torch.max(dx, dim=0).values}")
        self.l.info(f"dx mean: {torch.mean(dx, dim=0)}")
        self.l.info(f"dx abs max: {torch.max(torch.abs(dx), dim=0).values}")

        self.l.info(f"dx_scaled min: {torch.min(dx_scaled, dim=0).values}")
        self.l.info(f"dx_scaled max: {torch.max(dx_scaled, dim=0).values}")
        self.l.info(f"dx_scaled abs max: {torch.max(torch.abs(dx_scaled), dim=0).values}")

        self.l.info(f"mse_components: {mse_components}")
        self.l.info(f"loss_f: {loss_f}")

        self.l.info(f"Any NaN in dx? {torch.isnan(dx).any().item()}")
        self.l.info(f"Any Inf in dx? {torch.isinf(dx).any().item()}")
        self.l.info("================================================")

    @torch.no_grad()
    def debug_nearest_validation_state(self, u_query, y_hat_query=None):
        u_query = torch.tensor(
            u_query,
            dtype=torch.float32,
            device=self.device,
        )

        dist = torch.sum((self.u_data_val - u_query) ** 2, dim=1)
        idx = torch.argmin(dist)

        self.l.info("========== NEAREST VALIDATION STATE ==========")
        self.l.info(f"u_query: {u_query}")
        self.l.info(f"nearest u_val: {self.u_data_val[idx]}")
        self.l.info(f"nearest y_val: {self.y_data_val[idx]}")

        if y_hat_query is not None:
            y_hat_query = torch.tensor(
                y_hat_query,
                dtype=torch.float32,
                device=self.device,
            )
            self.l.info(f"y_hat query: {y_hat_query}")
            self.l.info(f"y_hat - y_val: {y_hat_query - self.y_data_val[idx]}")

        self.l.info(f"distance: {torch.sqrt(dist[idx])}")
        self.l.info("==============================================")

    @torch.no_grad()
    def debug_compare_predicted_vs_validation_algebraics(self, u_query, y_hat_query):
        """
        Compare algebraic quantities at the predicted worst point and at the
        nearest validation point.
        """

        self.net.eval()

        u_query = torch.tensor(
            u_query,
            dtype=torch.float32,
            device=self.device,
        )

        y_hat_query = torch.tensor(
            y_hat_query,
            dtype=torch.float32,
            device=self.device,
        )

        # Find nearest validation point
        dist = torch.sum((self.u_data_val - u_query) ** 2, dim=1)
        idx = torch.argmin(dist)

        u_val = self.u_data_val[idx]
        y_val = self.y_data_val[idx]

        # Add batch dimension
        u_hat_b = u_query.reshape(1, -1)
        y_hat_b = y_hat_query.reshape(1, -1)

        u_val_b = u_val.reshape(1, -1)
        y_val_b = y_val.reshape(1, -1)

        # Evaluate physics/algebraics at predicted point
        dx_hat, z_hat = self.physics_f(
            y_hat_b,
            u_hat_b,
            BSW=self.BSW,
            GOR=self.GOR,
            PI=self.PI,
            K_gs=self.K_gs,
            K_inj=self.K_inj,
            K_pr=self.K_pr,
            return_z=True,
        )

        # Evaluate physics/algebraics at nearest validation point
        dx_val, z_val = self.physics_f(
            y_val_b,
            u_val_b,
            BSW=self.BSW,
            GOR=self.GOR,
            PI=self.PI,
            K_gs=self.K_gs,
            K_inj=self.K_inj,
            K_pr=self.K_pr,
            return_z=True,
        )

        dx_hat = dx_hat[0]
        dx_val = dx_val[0]
        z_hat = z_hat[0]
        z_val = z_val[0]

        self.l.info("========== PREDICTED VS VALIDATION ALGEBRAICS ==========")
        self.l.info(f"u_query:        {u_query}")
        self.l.info(f"nearest u_val:  {u_val}")
        self.l.info(f"distance:       {torch.sqrt(dist[idx])}")

        self.l.info(f"y_hat:          {y_hat_query}")
        self.l.info(f"y_val:          {y_val}")
        self.l.info(f"y_hat - y_val:  {y_hat_query - y_val}")

        self.l.info(f"dx_hat:         {dx_hat}")
        self.l.info(f"dx_val:         {dx_val}")
        self.l.info(f"dx_hat-dx_val:  {dx_hat - dx_val}")

        self.l.info("--------------------------------------------------------")
        self.l.info(f"{'name':25s} {'pred':>15s} {'val':>15s} {'diff':>15s}")
        self.l.info("--------------------------------------------------------")

        for name, pred_value, val_value in zip(Z_DIAG_NAMES, z_hat, z_val):
            pred_float = float(pred_value.item())
            val_float = float(val_value.item())
            diff_float = pred_float - val_float

            self.l.info(
                f"{name:25s} "
                f"{pred_float:15.6e} "
                f"{val_float:15.6e} "
                f"{diff_float:15.6e}"
            )

        self.l.info("========================================================")

