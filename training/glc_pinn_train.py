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
        self.U1_MIN=0.10
        self.U2_MIN=0.20
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
        # self.poly_dataset_path = self.well_folder / "poly_dataset.pkl"
        self.validation_path = self.well_folder / "sweep_results_figures_training.pkl"
        self.training_path = self.well_folder / "sweep_results_validation.pkl"

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
        self.l.info(f"Loading the supervised data training dataset:")
        self.l.info(str(self.training_path))

        if not self.training_path.exists():
            raise FileNotFoundError(
                f"Could not find sweep_results.pkl for well {self.well_name}: "
                f"{self.training_path}"
            )

        with open(self.training_path, "rb") as f:
            sweep_results = pickle.load(f)

        output_names = ["m_G_an", "m_G_t", "m_o_t"]

        u_full_np, y_full_np = self._flatten_sweep_to_u_y(
            sweep_results,
            output_names,
        )

        self.train_output_names = list(output_names)

        self.l.info(
            f"Loaded full sweep dataset: "
            f"u_full={u_full_np.shape}, y_full={y_full_np.shape}"
        )

        self.u_data_train = torch.tensor(
            u_full_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_data_train = torch.tensor(
            y_full_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_data_train_norm = (self.y_data_train - self.y_min_t)/self.y_range_t

        # -----------------------------
        # 3) Physics collocation points
        # -----------------------------

        self.l.info(f"Generating {self.N_col} random physics collocation points.")
        np.random.seed(333333)

        u_col_extra_np = np.random.uniform(self.u_min_train,
                                     self.u_max_train,
                                     size=(self.N_col, self.N_u)
                                     ).astype(np.float32)

        self.u_col = torch.tensor(u_col_extra_np,
                                  dtype=torch.float32,
                                  device=self.device)

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
                          K_pr=self.K_pr)

        if isinstance(physics_out, tuple):
            dx = physics_out[0]
        else:
            dx = physics_out

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
        loss_data = torch.mean((y_pred_norm - self.y_data_train_norm) ** 2)
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

    # @torch.no_grad()
    # def print_prediction_debug(self, n_train=10, n_val=10, random_val=True):
    #     """
    #     Print ground-truth vs PINN prediction for selected training and validation points.
    #     Useful for checking scale/order/data consistency.
    #     """
    #
    #     self.net.eval()
    #
    #     output_names = getattr(
    #         self,
    #         "train_output_names",
    #         ["y0", "y1", "y2"],
    #     )
    #
    #     self.l.info("=" * 80)
    #     self.l.info(f"Prediction debug for well {self.well_name}")
    #     self.l.info(f"Output names: {output_names}")
    #     self.l.info(f"y_min_train: {self.y_min_train}")
    #     self.l.info(f"y_max_train: {self.y_max_train}")
    #
    #     if hasattr(self, "u_col_data"):
    #         self.l.info(f"u_col_data shape: {tuple(self.u_col_data.shape)}")
    #
    #     if hasattr(self, "u_col_extra"):
    #         self.l.info(f"u_col_extra shape: {tuple(self.u_col_extra.shape)}")
    #
    #     self.l.info("=" * 80)
    #
    #     # ------------------------------------------------
    #     # 1. Training points
    #     # ------------------------------------------------
    #     y_pred_train = self.net(self.u_data_train)
    #     err_train = y_pred_train - self.y_data_train
    #
    #     y_pred_train_norm = (y_pred_train - self.y_min_t) / self.y_range_t
    #     err_train_norm = y_pred_train_norm - self.y_data_train_norm
    #
    #     train_n = min(n_train, self.u_data_train.shape[0])
    #
    #     self.l.info("")
    #     self.l.info(f"First {train_n} training points:")
    #     self.l.info("-" * 80)
    #
    #     for k in range(train_n):
    #         u_k = self.u_data_train[k].detach().cpu().numpy()
    #         y_true_k = self.y_data_train[k].detach().cpu().numpy()
    #         y_pred_k = y_pred_train[k].detach().cpu().numpy()
    #         err_k = err_train[k].detach().cpu().numpy()
    #         err_norm_k = err_train_norm[k].detach().cpu().numpy()
    #
    #         self.l.info(
    #             f"TRAIN k={k:02d} | "
    #             f"u={u_k} | "
    #             f"true={y_true_k} | "
    #             f"pred={y_pred_k} | "
    #             f"err={err_k} | "
    #             f"err_norm={err_norm_k}"
    #         )
    #
    #     # ------------------------------------------------
    #     # 2. Validation points
    #     # ------------------------------------------------
    #     y_pred_val = self.net(self.u_data_val)
    #     err_val = y_pred_val - self.y_data_val
    #
    #     y_pred_val_norm = (y_pred_val - self.y_min_t) / self.y_range_t
    #     err_val_norm = y_pred_val_norm - self.y_data_val_norm
    #
    #     val_total = self.u_data_val.shape[0]
    #     val_n = min(n_val, val_total)
    #
    #     if random_val:
    #         rng = np.random.default_rng(333333)
    #         val_indices = rng.choice(val_total, size=val_n, replace=False)
    #     else:
    #         val_indices = np.linspace(0, val_total - 1, val_n, dtype=int)
    #
    #     self.l.info("")
    #     self.l.info(f"{val_n} validation points:")
    #     self.l.info("-" * 80)
    #
    #     for k in val_indices:
    #         u_k = self.u_data_val[k].detach().cpu().numpy()
    #         y_true_k = self.y_data_val[k].detach().cpu().numpy()
    #         y_pred_k = y_pred_val[k].detach().cpu().numpy()
    #         err_k = err_val[k].detach().cpu().numpy()
    #         err_norm_k = err_val_norm[k].detach().cpu().numpy()
    #
    #         self.l.info(
    #             f"VAL k={int(k):03d} | "
    #             f"u={u_k} | "
    #             f"true={y_true_k} | "
    #             f"pred={y_pred_k} | "
    #             f"err={err_k} | "
    #             f"err_norm={err_norm_k}"
    #         )
    #
    #     # ------------------------------------------------
    #     # 3. Summary statistics
    #     # ------------------------------------------------
    #     train_rmse = torch.sqrt(torch.mean(err_train ** 2, dim=0))
    #     train_maxabs = torch.max(torch.abs(err_train), dim=0).values
    #
    #     train_mse_norm = torch.mean(err_train_norm ** 2, dim=0)
    #
    #     val_rmse = torch.sqrt(torch.mean(err_val ** 2, dim=0))
    #     val_maxabs = torch.max(torch.abs(err_val), dim=0).values
    #
    #     val_mse_norm = torch.mean(err_val_norm ** 2, dim=0)
    #
    #     self.l.info("")
    #     self.l.info("Summary:")
    #     self.l.info("-" * 80)
    #     self.l.info(f"Train RMSE by component: {train_rmse.detach().cpu().numpy()}")
    #     self.l.info(f"Train MaxAbs by component: {train_maxabs.detach().cpu().numpy()}")
    #     self.l.info(f"Train normalized MSE by component: {train_mse_norm.detach().cpu().numpy()}")
    #
    #     self.l.info(f"Val RMSE by component: {val_rmse.detach().cpu().numpy()}")
    #     self.l.info(f"Val MaxAbs by component: {val_maxabs.detach().cpu().numpy()}")
    #     self.l.info(f"Val normalized MSE by component: {val_mse_norm.detach().cpu().numpy()}")
    #     self.l.info("=" * 80)
    # #
    # @torch.no_grad()
    # def print_physics_on_ground_truth_data(self):
    #     """
    #     Evaluate physics residual using the true 17 training states directly,
    #     and also compare with the network prediction at the same points.
    #     """
    #
    #     self.net.eval()
    #
    #     # ------------------------------------------------
    #     # Helper to handle physics_f returning dx or (dx, z)
    #     # ------------------------------------------------
    #     def _extract_dx(physics_out):
    #         if isinstance(physics_out, tuple):
    #             return physics_out[0]
    #         return physics_out
    #
    #     y_range = (self.y_max_t - self.y_min_t).clamp_min(1e-6)
    #
    #     # ------------------------------------------------
    #     # 1. Physics residual on TRUE 17 training states
    #     # ------------------------------------------------
    #     physics_out_true = self.physics_f(
    #         self.y_data_train,
    #         self.u_data_train,
    #         BSW=self.BSW,
    #         GOR=self.GOR,
    #         PI=self.PI,
    #         K_gs=self.K_gs,
    #         K_inj=self.K_inj,
    #         K_pr=self.K_pr,
    #     )
    #
    #     dx_true = _extract_dx(physics_out_true)
    #
    #     dx_true_scaled = dx_true / y_range
    #
    #     mse_true_components = torch.mean(dx_true_scaled ** 2, dim=0)
    #     mean_abs_true_raw = torch.mean(torch.abs(dx_true), dim=0)
    #     max_abs_true_raw = torch.max(torch.abs(dx_true), dim=0).values
    #
    #     # ------------------------------------------------
    #     # 2. Physics residual on NETWORK prediction at 17 training points
    #     # ------------------------------------------------
    #     y_pred_train = self.net(self.u_data_train)
    #
    #     physics_out_pred = self.physics_f(
    #         y_pred_train,
    #         self.u_data_train,
    #         BSW=self.BSW,
    #         GOR=self.GOR,
    #         PI=self.PI,
    #         K_gs=self.K_gs,
    #         K_inj=self.K_inj,
    #         K_pr=self.K_pr,
    #     )
    #
    #     dx_pred = _extract_dx(physics_out_pred)
    #
    #     dx_pred_scaled = dx_pred / y_range
    #
    #     mse_pred_components = torch.mean(dx_pred_scaled ** 2, dim=0)
    #     mean_abs_pred_raw = torch.mean(torch.abs(dx_pred), dim=0)
    #     max_abs_pred_raw = torch.max(torch.abs(dx_pred), dim=0).values
    #
    #     # ------------------------------------------------
    #     # 3. Optional: physics residual on extra collocation points
    #     # ------------------------------------------------
    #     extra_info = None
    #
    #     if hasattr(self, "u_col_extra"):
    #         y_pred_extra = self.net(self.u_col_extra)
    #
    #         physics_out_extra = self.physics_f(
    #             y_pred_extra,
    #             self.u_col_extra,
    #             BSW=self.BSW,
    #             GOR=self.GOR,
    #             PI=self.PI,
    #             K_gs=self.K_gs,
    #             K_inj=self.K_inj,
    #             K_pr=self.K_pr,
    #         )
    #
    #         dx_extra = _extract_dx(physics_out_extra)
    #
    #         dx_extra_scaled = dx_extra / y_range
    #
    #         extra_info = {
    #             "mse_components": torch.mean(dx_extra_scaled ** 2, dim=0),
    #             "mean_abs_raw": torch.mean(torch.abs(dx_extra), dim=0),
    #             "max_abs_raw": torch.max(torch.abs(dx_extra), dim=0).values,
    #         }
    #
    #     # ------------------------------------------------
    #     # Logging
    #     # ------------------------------------------------
    #     self.l.info("=" * 80)
    #     self.l.info("Physics residual diagnostics")
    #     self.l.info("-" * 80)
    #
    #     self.l.info("Physics residual on TRUE 17 training states:")
    #     self.l.info(f"Raw mean abs dx: {mean_abs_true_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Raw max abs dx:  {max_abs_true_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Scaled MSE components: {mse_true_components.detach().cpu().numpy()}")
    #
    #     self.l.info("-" * 80)
    #     self.l.info("Physics residual on PINN prediction at 17 training points:")
    #     self.l.info(f"Raw mean abs dx: {mean_abs_pred_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Raw max abs dx:  {max_abs_pred_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Scaled MSE components: {mse_pred_components.detach().cpu().numpy()}")
    #
    #     if extra_info is not None:
    #         self.l.info("-" * 80)
    #         self.l.info("Physics residual on PINN prediction at extra collocation points:")
    #         self.l.info(f"Raw mean abs dx: {extra_info['mean_abs_raw'].detach().cpu().numpy()}")
    #         self.l.info(f"Raw max abs dx:  {extra_info['max_abs_raw'].detach().cpu().numpy()}")
    #         self.l.info(f"Scaled MSE components: {extra_info['mse_components'].detach().cpu().numpy()}")
    #
    #     self.l.info("=" * 80)


    # @torch.no_grad()
    # def _log_epoch_info(self, train_losses: dict, val_losses: dict):
    #     """
    #     Print compact epoch information during training.
    #
    #     train_losses comes from train_pass().
    #     val_losses comes from validation_pass().
    #     """
    #
    #     self.net.eval()
    #
    #     y_probe = self.net(self._u_probe_t).detach().cpu().numpy().reshape(-1)
    #
    #     output_names = getattr(
    #         self,
    #         "train_output_names",
    #         ["y0", "y1", "y2"],
    #     )
    #
    #     self.l.info(
    #         f"Epoch {self._e:6d} | "
    #         f"train total={train_losses['total']:.4e} "
    #         f"(phys={train_losses['physics']:.3e}, "
    #         f"data_n={train_losses['data']:.3e}) | "
    #         f"f=[{train_losses['physics_comp_0']:.3e}, "
    #         f"{train_losses['physics_comp_1']:.3e}, "
    #         f"{train_losses['physics_comp_2']:.3e}] | "
    #         f"val mse={val_losses['total']:.4e}, "
    #         f"val rmse={val_losses['rmse_total']:.4e} | "
    #         f"val rmse comps="
    #         f"[{val_losses['rmse_y0']:.3e}, "
    #         f"{val_losses['rmse_y1']:.3e}, "
    #         f"{val_losses['rmse_y2']:.3e}] | "
    #         f"val maxabs comps="
    #         f"[{val_losses['max_abs_y0']:.3e}, "
    #         f"{val_losses['max_abs_y1']:.3e}, "
    #         f"{val_losses['max_abs_y2']:.3e}] | "
    #         f"u_probe={tuple(self.u_probe.tolist())} "
    #         f"y_hat={y_probe} "
    #         f"outputs={output_names}"
    #     )

    # @torch.no_grad()
    # def print_prediction_debug(self, n_train=10, n_val=10, random_val=True):
    #     """
    #     Print ground-truth vs PINN prediction for selected training and validation points.
    #     Useful for checking scale/order/data consistency.
    #     """
    #
    #     self.net.eval()
    #
    #     output_names = getattr(
    #         self,
    #         "train_output_names",
    #         ["y0", "y1", "y2"],
    #     )
    #
    #     self.l.info("=" * 80)
    #     self.l.info(f"Prediction debug for well {self.well_name}")
    #     self.l.info(f"Output names: {output_names}")
    #     self.l.info(f"y_min_train: {self.y_min_train}")
    #     self.l.info(f"y_max_train: {self.y_max_train}")
    #     self.l.info("=" * 80)
    #
    #     # ------------------------------------------------
    #     # 1. Training points
    #     # ------------------------------------------------
    #     y_pred_train = self.net(self.u_data_train)
    #     err_train = y_pred_train - self.y_data_train
    #
    #     train_n = min(n_train, self.u_data_train.shape[0])
    #
    #     self.l.info("")
    #     self.l.info(f"First {train_n} training points:")
    #     self.l.info("-" * 80)
    #
    #     for k in range(train_n):
    #         u_k = self.u_data_train[k].detach().cpu().numpy()
    #         y_true_k = self.y_data_train[k].detach().cpu().numpy()
    #         y_pred_k = y_pred_train[k].detach().cpu().numpy()
    #         err_k = err_train[k].detach().cpu().numpy()
    #
    #         self.l.info(
    #             f"TRAIN k={k:02d} | "
    #             f"u={u_k} | "
    #             f"true={y_true_k} | "
    #             f"pred={y_pred_k} | "
    #             f"err={err_k}"
    #         )
    #
    #     # ------------------------------------------------
    #     # 2. Validation points
    #     # ------------------------------------------------
    #     y_pred_val = self.net(self.u_data_val)
    #     err_val = y_pred_val - self.y_data_val
    #
    #     val_total = self.u_data_val.shape[0]
    #     val_n = min(n_val, val_total)
    #
    #     if random_val:
    #         rng = np.random.default_rng(333333)
    #         val_indices = rng.choice(val_total, size=val_n, replace=False)
    #     else:
    #         val_indices = np.linspace(0, val_total - 1, val_n, dtype=int)
    #
    #     self.l.info("")
    #     self.l.info(f"{val_n} validation points:")
    #     self.l.info("-" * 80)
    #
    #     for k in val_indices:
    #         u_k = self.u_data_val[k].detach().cpu().numpy()
    #         y_true_k = self.y_data_val[k].detach().cpu().numpy()
    #         y_pred_k = y_pred_val[k].detach().cpu().numpy()
    #         err_k = err_val[k].detach().cpu().numpy()
    #
    #         self.l.info(
    #             f"VAL k={int(k):03d} | "
    #             f"u={u_k} | "
    #             f"true={y_true_k} | "
    #             f"pred={y_pred_k} | "
    #             f"err={err_k}"
    #         )
    #
    #     # ------------------------------------------------
    #     # 3. Summary statistics
    #     # ------------------------------------------------
    #     train_rmse = torch.sqrt(torch.mean(err_train ** 2, dim=0))
    #     train_maxabs = torch.max(torch.abs(err_train), dim=0).values
    #
    #     val_rmse = torch.sqrt(torch.mean(err_val ** 2, dim=0))
    #     val_maxabs = torch.max(torch.abs(err_val), dim=0).values
    #
    #     self.l.info("")
    #     self.l.info("Summary:")
    #     self.l.info("-" * 80)
    #     self.l.info(f"Train RMSE by component: {train_rmse.detach().cpu().numpy()}")
    #     self.l.info(f"Train MaxAbs by component: {train_maxabs.detach().cpu().numpy()}")
    #     self.l.info(f"Val RMSE by component: {val_rmse.detach().cpu().numpy()}")
    #     self.l.info(f"Val MaxAbs by component: {val_maxabs.detach().cpu().numpy()}")
    #     self.l.info("=" * 80)
    #
    # @torch.no_grad()
    # def print_physics_on_ground_truth_data(self):
    #     """
    #     Evaluate physics residual using the true 17 training states directly,
    #     not the neural network prediction.
    #     """
    #
    #     dx_data = self.physics_f(
    #         self.y_data_train,
    #         self.u_data_train,
    #         BSW=self.BSW,
    #         GOR=self.GOR,
    #         PI=self.PI,
    #         K_gs=self.K_gs,
    #         K_inj=self.K_inj,
    #         K_pr=self.K_pr,
    #     )
    #
    #     y_range = (self.y_max_t - self.y_min_t).clamp_min(1e-6)
    #     dx_scaled = dx_data / y_range
    #
    #     mse_components = torch.mean(dx_scaled ** 2, dim=0)
    #     mean_abs_raw = torch.mean(torch.abs(dx_data), dim=0)
    #     max_abs_raw = torch.max(torch.abs(dx_data), dim=0).values
    #
    #     self.l.info("=" * 80)
    #     self.l.info("Physics residual on TRUE 17 training states")
    #     self.l.info("-" * 80)
    #     self.l.info(f"Raw mean abs dx: {mean_abs_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Raw max abs dx:  {max_abs_raw.detach().cpu().numpy()}")
    #     self.l.info(f"Scaled MSE components: {mse_components.detach().cpu().numpy()}")
    #     self.l.info("=" * 80)
    #
  # def print_flow_balance_diagnostics(self, label=""):
  #       """
  #       Detailed diagnostic for the steady-state mass-balance residuals.
  #
  #       For a fixed set of diagnostic training points, this compares:
  #           1. the true sweep state y_data_train
  #           2. the PINN prediction net(u)
  #
  #       It prints:
  #           - state errors
  #           - raw residuals dx
  #           - the individual flow terms that form dx1, dx2, dx3
  #           - pressure drops and phase-fraction variables that may explain blow-ups
  #
  #       Required:
  #           physics_f(..., return_z=True) must return:
  #               dx, z_diag
  #
  #           where z_diag follows self.Z_DIAG_NAMES.
  #       """
  #
  #       self.net.eval()
  #
  #       if not hasattr(self, "diag_indices"):
  #           raise RuntimeError(
  #               "self.diag_indices not found. Create it in prepare_data()."
  #           )
  #
  #       if not hasattr(self, "z_diag_idx"):
  #           raise RuntimeError(
  #               "self.z_diag_idx not found. Define self.Z_DIAG_NAMES and self.z_diag_idx."
  #           )
  #
  #       idx = self.diag_indices
  #
  #       u = self.u_data_train[idx]
  #       y_true = self.y_data_train[idx]
  #       y_pred = self.net(u)
  #
  #       # ------------------------------------------------
  #       # Evaluate physics on true states and predicted states
  #       # ------------------------------------------------
  #       dx_true, z_true = self._eval_physics_dx_z(y_true, u)
  #       dx_pred, z_pred = self._eval_physics_dx_z(y_pred, u)
  #
  #       zi = self.z_diag_idx
  #
  #       def get(z, name):
  #           return z[:, zi[name]]
  #
  #       def scalar(t, k):
  #           return float(t[k].detach().cpu().item())
  #
  #       def arr(t, k):
  #           return t[k].detach().cpu().numpy()
  #
  #       # ------------------------------------------------
  #       # Print header
  #       # ------------------------------------------------
  #       self.l.info("=" * 120)
  #       self.l.info(f"FLOW BALANCE DIAGNOSTICS {label} | epoch={self._e}")
  #       self.l.info("-" * 120)
  #       self.l.info(
  #           "Residual definitions: "
  #           "dx1 = w_G_in - w_G_inj, "
  #           "dx2 = w_G_inj + w_G_res - w_G_out, "
  #           "dx3 = w_L_res - w_L_out"
  #       )
  #       self.l.info("-" * 120)
  #
  #       # ------------------------------------------------
  #       # Loop over diagnostic points
  #       # ------------------------------------------------
  #       for local_k in range(idx.shape[0]):
  #           global_k = int(idx[local_k].detach().cpu().item())
  #
  #           u_k = arr(u, local_k)
  #           y_true_k = arr(y_true, local_k)
  #           y_pred_k = arr(y_pred, local_k)
  #           y_err_k = y_pred_k - y_true_k
  #
  #           dx_true_k = arr(dx_true, local_k)
  #           dx_pred_k = arr(dx_pred, local_k)
  #
  #           self.l.info("")
  #           self.l.info("=" * 100)
  #           self.l.info(f"POINT global_k={global_k} | u={u_k}")
  #           self.l.info("-" * 100)
  #
  #           # ------------------------------------------------
  #           # State comparison
  #           # ------------------------------------------------
  #           self.l.info("STATE COMPARISON")
  #           self.l.info(f"  y_true = {y_true_k}")
  #           self.l.info(f"  y_pred = {y_pred_k}")
  #           self.l.info(f"  y_err  = {y_err_k}")
  #
  #           # ------------------------------------------------
  #           # Residual summary
  #           # ------------------------------------------------
  #           self.l.info("")
  #           self.l.info("RESIDUAL SUMMARY")
  #           self.l.info(
  #               f"  TRUE dx = {dx_true_k} | norm={np.linalg.norm(dx_true_k):.6e}"
  #           )
  #           self.l.info(
  #               f"  PRED dx = {dx_pred_k} | norm={np.linalg.norm(dx_pred_k):.6e}"
  #           )
  #
  #           # ------------------------------------------------
  #           # TRUE balance terms
  #           # ------------------------------------------------
  #           self.l.info("")
  #           self.l.info("TRUE FLOW BALANCES")
  #
  #           self.l.info(
  #               "  dx1 = w_G_in - w_G_inj = "
  #               f"{scalar(get(z_true, 'w_G_in'), local_k):.6e} - "
  #               f"{scalar(get(z_true, 'w_G_inj'), local_k):.6e} = "
  #               f"{float(dx_true[local_k, 0].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  dx2 = w_G_inj + w_G_res - w_G_out = "
  #               f"{scalar(get(z_true, 'w_G_inj'), local_k):.6e} + "
  #               f"{scalar(get(z_true, 'w_G_res'), local_k):.6e} - "
  #               f"{scalar(get(z_true, 'w_G_out'), local_k):.6e} = "
  #               f"{float(dx_true[local_k, 1].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  dx3 = w_L_res - w_L_out = "
  #               f"{scalar(get(z_true, 'w_L_res'), local_k):.6e} - "
  #               f"{scalar(get(z_true, 'w_L_out'), local_k):.6e} = "
  #               f"{float(dx_true[local_k, 2].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  flows: "
  #               f"w_res={scalar(get(z_true, 'w_res'), local_k):.6e}, "
  #               f"w_out={scalar(get(z_true, 'w_out'), local_k):.6e}, "
  #               f"w_G_in={scalar(get(z_true, 'w_G_in'), local_k):.6e}, "
  #               f"w_G_inj={scalar(get(z_true, 'w_G_inj'), local_k):.6e}, "
  #               f"w_G_res={scalar(get(z_true, 'w_G_res'), local_k):.6e}, "
  #               f"w_L_res={scalar(get(z_true, 'w_L_res'), local_k):.6e}, "
  #               f"w_G_out={scalar(get(z_true, 'w_G_out'), local_k):.6e}, "
  #               f"w_L_out={scalar(get(z_true, 'w_L_out'), local_k):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  pressures: "
  #               f"P_bh={scalar(get(z_true, 'P_bh_bar'), local_k):.6e} bar, "
  #               f"P_tb_b={scalar(get(z_true, 'P_tb_b_bar'), local_k):.6e} bar, "
  #               f"P_tb_t={scalar(get(z_true, 'P_tb_t_bar'), local_k):.6e} bar, "
  #               f"P_an_t={scalar(get(z_true, 'P_an_t_bar'), local_k):.6e} bar, "
  #               f"P_an_b={scalar(get(z_true, 'P_an_b_bar'), local_k):.6e} bar"
  #           )
  #
  #           self.l.info(
  #               "  deltas: "
  #               f"dP_gs_an={scalar(get(z_true, 'dP_gs_an_bar'), local_k):.6e} bar, "
  #               f"dP_an_tb={scalar(get(z_true, 'dP_an_tb_bar'), local_k):.6e} bar, "
  #               f"dP_res_bh={scalar(get(z_true, 'dP_res_bh_bar'), local_k):.6e} bar, "
  #               f"dP_tb_choke={scalar(get(z_true, 'dP_tb_choke_bar'), local_k):.6e} bar"
  #           )
  #
  #           self.l.info(
  #               "  phase/mixture: "
  #               f"alpha_avg_L_tb={scalar(get(z_true, 'alpha_avg_L_tb'), local_k):.6e}, "
  #               f"alpha_L_tb_b={scalar(get(z_true, 'alpha_L_tb_b'), local_k):.6e}, "
  #               f"alpha_L_tb_t={scalar(get(z_true, 'alpha_L_tb_t'), local_k):.6e}, "
  #               f"alpha_G_tb_t={scalar(get(z_true, 'alpha_G_tb_t'), local_k):.6e}, "
  #               f"rho_mix_tb_t={scalar(get(z_true, 'rho_mix_tb_t'), local_k):.6e}, "
  #               f"rho_mix_tb_t_safe={scalar(get(z_true, 'rho_mix_tb_t_safe'), local_k):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  denominators: "
  #               f"V_gas_tb_t={scalar(get(z_true, 'V_gas_tb_t'), local_k):.6e}, "
  #               f"V_gas_tb_t_safe={scalar(get(z_true, 'V_gas_tb_t_safe'), local_k):.6e}, "
  #               f"denom_alpha_t={scalar(get(z_true, 'denom_alpha_t'), local_k):.6e}, "
  #               f"denom_alpha_t_safe={scalar(get(z_true, 'denom_alpha_t_safe'), local_k):.6e}, "
  #               f"denom_alpha_b={scalar(get(z_true, 'denom_alpha_b'), local_k):.6e}, "
  #               f"denom_alpha_b_safe={scalar(get(z_true, 'denom_alpha_b_safe'), local_k):.6e}"
  #           )
  #
  #           # ------------------------------------------------
  #           # PRED balance terms
  #           # ------------------------------------------------
  #           self.l.info("")
  #           self.l.info("PRED FLOW BALANCES")
  #
  #           self.l.info(
  #               "  dx1 = w_G_in - w_G_inj = "
  #               f"{scalar(get(z_pred, 'w_G_in'), local_k):.6e} - "
  #               f"{scalar(get(z_pred, 'w_G_inj'), local_k):.6e} = "
  #               f"{float(dx_pred[local_k, 0].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  dx2 = w_G_inj + w_G_res - w_G_out = "
  #               f"{scalar(get(z_pred, 'w_G_inj'), local_k):.6e} + "
  #               f"{scalar(get(z_pred, 'w_G_res'), local_k):.6e} - "
  #               f"{scalar(get(z_pred, 'w_G_out'), local_k):.6e} = "
  #               f"{float(dx_pred[local_k, 1].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  dx3 = w_L_res - w_L_out = "
  #               f"{scalar(get(z_pred, 'w_L_res'), local_k):.6e} - "
  #               f"{scalar(get(z_pred, 'w_L_out'), local_k):.6e} = "
  #               f"{float(dx_pred[local_k, 2].detach().cpu().item()):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  flows: "
  #               f"w_res={scalar(get(z_pred, 'w_res'), local_k):.6e}, "
  #               f"w_out={scalar(get(z_pred, 'w_out'), local_k):.6e}, "
  #               f"w_G_in={scalar(get(z_pred, 'w_G_in'), local_k):.6e}, "
  #               f"w_G_inj={scalar(get(z_pred, 'w_G_inj'), local_k):.6e}, "
  #               f"w_G_res={scalar(get(z_pred, 'w_G_res'), local_k):.6e}, "
  #               f"w_L_res={scalar(get(z_pred, 'w_L_res'), local_k):.6e}, "
  #               f"w_G_out={scalar(get(z_pred, 'w_G_out'), local_k):.6e}, "
  #               f"w_L_out={scalar(get(z_pred, 'w_L_out'), local_k):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  pressures: "
  #               f"P_bh={scalar(get(z_pred, 'P_bh_bar'), local_k):.6e} bar, "
  #               f"P_tb_b={scalar(get(z_pred, 'P_tb_b_bar'), local_k):.6e} bar, "
  #               f"P_tb_t={scalar(get(z_pred, 'P_tb_t_bar'), local_k):.6e} bar, "
  #               f"P_an_t={scalar(get(z_pred, 'P_an_t_bar'), local_k):.6e} bar, "
  #               f"P_an_b={scalar(get(z_pred, 'P_an_b_bar'), local_k):.6e} bar"
  #           )
  #
  #           self.l.info(
  #               "  deltas: "
  #               f"dP_gs_an={scalar(get(z_pred, 'dP_gs_an_bar'), local_k):.6e} bar, "
  #               f"dP_an_tb={scalar(get(z_pred, 'dP_an_tb_bar'), local_k):.6e} bar, "
  #               f"dP_res_bh={scalar(get(z_pred, 'dP_res_bh_bar'), local_k):.6e} bar, "
  #               f"dP_tb_choke={scalar(get(z_pred, 'dP_tb_choke_bar'), local_k):.6e} bar"
  #           )
  #
  #           self.l.info(
  #               "  phase/mixture: "
  #               f"alpha_avg_L_tb={scalar(get(z_pred, 'alpha_avg_L_tb'), local_k):.6e}, "
  #               f"alpha_L_tb_b={scalar(get(z_pred, 'alpha_L_tb_b'), local_k):.6e}, "
  #               f"alpha_L_tb_t={scalar(get(z_pred, 'alpha_L_tb_t'), local_k):.6e}, "
  #               f"alpha_G_tb_t={scalar(get(z_pred, 'alpha_G_tb_t'), local_k):.6e}, "
  #               f"rho_mix_tb_t={scalar(get(z_pred, 'rho_mix_tb_t'), local_k):.6e}, "
  #               f"rho_mix_tb_t_safe={scalar(get(z_pred, 'rho_mix_tb_t_safe'), local_k):.6e}"
  #           )
  #
  #           self.l.info(
  #               "  denominators: "
  #               f"V_gas_tb_t={scalar(get(z_pred, 'V_gas_tb_t'), local_k):.6e}, "
  #               f"V_gas_tb_t_safe={scalar(get(z_pred, 'V_gas_tb_t_safe'), local_k):.6e}, "
  #               f"denom_alpha_t={scalar(get(z_pred, 'denom_alpha_t'), local_k):.6e}, "
  #               f"denom_alpha_t_safe={scalar(get(z_pred, 'denom_alpha_t_safe'), local_k):.6e}, "
  #               f"denom_alpha_b={scalar(get(z_pred, 'denom_alpha_b'), local_k):.6e}, "
  #               f"denom_alpha_b_safe={scalar(get(z_pred, 'denom_alpha_b_safe'), local_k):.6e}"
  #           )
  #
  #       self.l.info("=" * 120)

# def get_staged_physics_losses(self):
#     """
#     Computes the physics losses needed by the curriculum.
#
#     Returns
#     -------
#     loss_f_data : physics loss on the 17 supervised points
#     mse_f_data  : component-wise physics MSE on the 17 points
#     loss_f_extra: physics loss on the additional collocation points
#     mse_f_extra : component-wise physics MSE on the extra points
#     """
#
#     loss_f_data, mse_f_data, y_hat_data, dx_data = self.get_physics_loss(
#         self.u_col_data
#     )
#
#     loss_f_extra, mse_f_extra, y_hat_extra, dx_extra = self.get_physics_loss(
#         self.u_col_extra
#     )
#
#     return {
#         "loss_f_data": loss_f_data,
#         "mse_f_data": mse_f_data,
#         "y_hat_data": y_hat_data,
#         "dx_data": dx_data,
#
#         "loss_f_extra": loss_f_extra,
#         "mse_f_extra": mse_f_extra,
#         "y_hat_extra": y_hat_extra,
#         "dx_extra": dx_extra,
# #     }
# @torch.no_grad()
    # def _eval_physics_dx_z(self, y, u):
    #     """
    #     Evaluate physics function and return dx, z.
    #     Assumes physics_f supports return_z=True.
    #     """
    #     physics_out = self.physics_f(
    #         y,
    #         u,
    #         BSW=self.BSW,
    #         GOR=self.GOR,
    #         PI=self.PI,
    #         K_gs=self.K_gs,
    #         K_inj=self.K_inj,
    #         K_pr=self.K_pr,
    #         return_z=True,
    #     )
    #
    #     if not isinstance(physics_out, tuple):
    #         raise RuntimeError(
    #             "Expected physics_f(..., return_z=True) to return (dx, z), "
    #             "but it returned only dx."
    #         )
    #
    #     dx, z = physics_out
    #     return dx, z
    #

# u1_grid = np.asarray(sweep_results["u1_grid"], dtype=float)
# u2_grid = np.asarray(sweep_results["u2_grid"], dtype=float)
#
# u_list = []
# y_list = []
#
# for i, u1 in enumerate(u1_grid):
#     for j, u2 in enumerate(u2_grid):
#
#         y_row = [
#             sweep_results["OUT"][name][i, j]
#             for name in output_names
#         ]
#
#         y_row = np.asarray(y_row, dtype=float)
#
#         if np.all(np.isfinite(y_row)):
#             u_list.append([u1, u2])
#             y_list.append(y_row)
#
# u_full_np = np.asarray(u_list, dtype=np.float32)
# y_full_np = np.asarray(y_list, dtype=np.float32)
#
# if u_full_np.shape[1] != self.N_u:
#     raise ValueError(
#         f"Expected u_full_np to have {self.N_u} columns, "
#         f"but got shape {u_full_np.shape}."
#     )
#
# if y_full_np.shape[1] != self.N_y:
#     raise ValueError(
#         f"Expected y_full_np to have {self.N_y} columns, "
#         f"but got shape {y_full_np.shape}."
#     )