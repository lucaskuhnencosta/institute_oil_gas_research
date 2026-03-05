import torch
import wandb
import numpy as np
from Networks.networks import PINN
from contextlib import nullcontext
from Surrogate_ODE_Model.glc_surrogate_torch import glc_surrogate_dx_torch_nostuck
from Training.base_trainer import Trainer
from Application.model_analysis_application import make_model
from Application.model_analysis_application import run_sweep
from Application.dataset_generator import flatten_sweep_results_to_batch


class SteadyStatePINNTrainer(Trainer):
    def __init__(
            self,
            net: PINN,
            surrogate_function,
            N_col: int = 5000,
            N_data: int = 30,
            lambda_data: float = 1000,
            adam_epochs: int = 5000,
            lbfgs_epochs: int = 2000,
            lr: float = 1e-3,
            y_min: list = [3032.55, 220.05, 6341.30],
            y_max: list = [4796.20, 1094.60, 11990.90],
            u_min: list = [0.05, 0.10],
            u_max: list = [1.0, 1.0],
            mse_f_scale_factors: list=[1.0, 1.0, 1.0],
            u_probe=(0.6, 0.6),
            mixed_precision=True,
            device=None,
            wandb_project="PINC-GasLift",
            wandb_group=None,
            random_seed=333333):

        self.N_y=3
        self.N_u=2

        self.lambda_data = lambda_data

        self.K1=adam_epochs
        self.K2=lbfgs_epochs

        super().__init__(
            net=net,
            epochs=adam_epochs+lbfgs_epochs,
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

        self.u_probe=np.array(u_probe)

        self.N_data=N_data
        self.N_col = N_col

        self.y_min_train = np.array(y_min)
        self.y_max_train = np.array(y_max)
        self.u_min_train = np.array(u_min)
        self.u_max_train = np.array(u_max)

        self.u1_min,self.u1_max = self.u_min_train[0], self.u_max_train[0]
        self.u2_min,self.u2_max = self.u_min_train[1], self.u_max_train[1]

        self.y_guess_surr=np.array([3285.42, 300.822, 6910.91], dtype=float)

        self.mse_f_scale = torch.tensor(mse_f_scale_factors, device=self.device)

        self.physics_f=surrogate_function

        self._add_to_wandb_config({
            "N_col": self.N_col,
            "N_data": self.N_data,
            'K1_adam': self.K1,
            'K2_lbfgs': self.K2,
            'y_min': y_min,
            'y_max': y_max,
            'u_min': u_min,
            'u_max': u_max,
            "mse_f_scale_factors": mse_f_scale_factors,
        })

    def prepare_data(self):
        # -----------------------------
        # 1) Physics collocation points
        # -----------------------------

        self.l.info("Preparing PINN data for Gas-Lift Well...")

        self.l.info(f"Generating {self.N_col} initial condition points...")
        np.random.seed(333333)

        u_col_np = np.random.uniform(self.u_min_train, self.u_max_train, size=(self.N_col, self.N_u)).astype(np.float32)
        self.u_col = torch.tensor(u_col_np, dtype=torch.float32, device=self.device)

        self._u_probe_t = torch.tensor(self.u_probe[None, :], dtype=torch.float32, device=self.device)

        # torch versions of y_min/y_max for normalization on device
        self.y_min_t = torch.tensor(self.y_min_train, dtype=torch.float32, device=self.device)
        self.y_max_t = torch.tensor(self.y_max_train, dtype=torch.float32, device=self.device)
        self.y_range_t = (self.y_max_t - self.y_min_t).clamp_min(1e-6)

        self.l.info(
            f"Prepared data: u_col={tuple(self.u_col.shape)} "
            f"u_probe={tuple(self.u_probe.tolist())}"
        )

        # -----------------------------
        # 2) Supervised sweep dataset
        # -----------------------------

        self.l.info(f"Generating supervised dataset via sweep (N_data={self.N_data} => {self.N_data ** 2} grid pts)...")

        u1_grid_train = np.linspace(float(self.u_min_train[0]), float(self.u_max_train[0]) + 1e-5, int(self.N_data))
        u2_grid_train = np.linspace(float(self.u_min_train[1]), float(self.u_max_train[1]) + 1e-5, int(self.N_data))

        model_sur = make_model("surrogate", BSW=0.20, GOR=0.05, PI=3.0e-6)

        results_train = run_sweep(
            model_sur,
            u1_grid=u1_grid_train,
            u2_grid=u2_grid_train,
            y_guess_init=self.y_guess_surr,
            z_guess_init=None,
        )

        batch_train = flatten_sweep_results_to_batch(results_train, only_success=True)

        # Move to device
        self.u_data_train = batch_train["u_t"].to(self.device)
        self.y_data_train = batch_train["y_t"].to(self.device)

        # ---------------------------------
        # 3) Supervised sweep dataset (VAL)
        # ---------------------------------
        N_data_val = int(self.N_data / 3)
        N_data_val = max(3, N_data_val)  # keep a minimum grid so it isn't degenerate

        self.l.info(f"Generating VAL sweep dataset with N_data_val={N_data_val} (grid {N_data_val}x{N_data_val}) ...")

        u1_grid_val = np.linspace(self.u_min_train[0], self.u_max_train[0] + 1e-5, N_data_val, dtype=float)
        u2_grid_val = np.linspace(self.u_min_train[1], self.u_max_train[1] + 1e-5, N_data_val, dtype=float)

        results_val = run_sweep(
            model_sur,
            u1_grid_val,
            u2_grid_val,
            y_guess_init=self.y_guess_surr,
            z_guess_init=None
        )

        batch_val= flatten_sweep_results_to_batch(results_val, only_success=True)

        # Move to device
        self.u_data_val = batch_val["u_t"].to(self.device)
        self.y_data_val = batch_val["y_t"].to(self.device)

        self.y_data_train_norm = (self.y_data_train - self.y_min_t) / (self.y_range_t)
        self.print_shapes()


    def print_shapes(self):
        self.l.info(f"Size of self y data train is:{self.y_data_train.shape}")
        self.l.info(f"Size of self u data train is:{self.u_data_train.shape}")

        self.l.info(f"Size of self y data val is:{self.y_data_val.shape}")

        self.l.info(f"Size of the colocation poins is {self.u_col.shape}")

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
                lr=0.05,
                max_iter=max_iter_per_call,
                line_search_fn="strong_wolfe"
            )
            self._add_to_wandb_config({
                "switched_to_lbfgs_at_epoch": self._e,
                "lbfgs_max_iter_per_call": max_iter_per_call
            })

    def get_physics_loss(self, u: torch.Tensor):
        y_hat=self.net(u)
        dx=self.physics_f(y_hat,u)

        # component-wise MSE

        # (1) scale per equation
        y_range = (self.net.y_max - self.net.y_min)  # (3,)
        dx_scaled = dx / y_range

        # (2) mse per component in scaled units
        mse_components = torch.mean(dx_scaled * dx_scaled, dim=0)  # (3,)

        # (3) optionally keep extra weights, but now they can be ~O(1)
        loss_f = torch.mean(self.mse_f_scale * mse_components)

        return loss_f, mse_components, y_hat, dx

    def _data_loss_raw(self):
        y_pred = self.net(self.u_data_val)
        return torch.mean((y_pred - self.y_data_val) ** 2), y_pred

    def _data_loss(self):
        y_pred=self.net(self.u_data_train)
        y_pred_n = (y_pred - self.y_min_t) / self.y_range_t

        return torch.mean((y_pred_n - self.y_data_train_norm) ** 2), y_pred_n

    def train_pass(self):
        self.net.train()

        if self.optimizer == 'LBFGS':
            def closure():
                self._optim.zero_grad(set_to_none=True)
                with self.autocast_if_mp():
                    loss_f, mse_f_comps, y_hat_col, dx_col = self.get_physics_loss(self.u_col)
                    loss_data_n, y_pred_data = self._data_loss()

                    loss = loss_f + self.lambda_data * loss_data_n
                loss.backward()
                #self._print_one_time_sanity(y_hat_col, dx_col, y_pred_data, loss_data_n, loss_data_raw)
                self._current_losses = {
                    'total': loss.item(),
                    "physics": loss_f.item(),
                    "data": float(loss_data_n.item()),
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
            loss_f, mse_f_comps,y_hat_col,dx_col = self.get_physics_loss(self.u_col)
            loss_data_n,y_pred_data = self._data_loss()
            if self._e<1000:
                loss = self.lambda_data * loss_data_n
            else:
                loss = loss_f + self.lambda_data * loss_data_n
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            loss.backward()
            # # after loss.backward()
            # with torch.no_grad():
            #     # 1) grad norm
            #     g2 = 0.0
            #     for p in self.net.parameters():
            #         if p.grad is not None:
            #             g2 += p.grad.detach().float().norm().item() ** 2
            #     grad_norm = g2 ** 0.5
            #
            #     # 2) step size proxy: parameter norm change if we took a step (approx)
            #     lr = self._optim.param_groups[0]["lr"]
            #     self.l.info(f"[DBG] epoch={self._e} grad_norm={grad_norm:.3e} lr={lr:.3e}")
            self._optim.step()
            # with torch.no_grad():
            #     w2 = 0.0
            #     for p in self.net.parameters():
            #         w2 += p.detach().float().norm().item() ** 2
            #     w_norm = w2 ** 0.5
            #     self.l.info(f"[DBG] epoch={self._e} w_norm={w_norm:.6e}")

        if self.lr_scheduler is not None:
            self._scheduler.step()
        return {
            "total": float(loss.item()),
            "physics": float(loss_f.item()),
            "data": float(loss_data_n.item()),
            "physics_comp_0": float(mse_f_comps[0].item()),
            "physics_comp_1": float(mse_f_comps[1].item()),
            "physics_comp_2": float(mse_f_comps[2].item()),
        }

    @torch.no_grad()
    def validation_pass(self):
        self.net.eval()
        with self.autocast_if_mp():
            loss_data_raw, _ = self._data_loss_raw()  # raw (physical units)

            loss = loss_data_raw
        return {
            "total": float(loss_data_raw.item()),
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
            f"(phys={train_losses['physics']:.3e}, data_n={train_losses['data']:.3e}) "
            f"f=[{train_losses['physics_comp_0']:.3e}, {train_losses['physics_comp_1']:.3e}, {train_losses['physics_comp_2']:.3e}] | "
            f"val data_raw(total)={val_losses['total']:.4e} | "
            f"u_probe={tuple(self.u_probe.tolist())} y_hat={y_probe}"
        )


    # def _debug_grad_norm(self):
    #     total = 0.0
    #     count = 0
    #     max_g = 0.0
    #     for p in self.net.parameters():
    #         if p.grad is None:
    #             continue
    #         g = p.grad.detach()
    #         n = torch.norm(g).item()
    #         total += n
    #         max_g = max(max_g, g.abs().max().item())
    #         count += 1
    #     return {"grad_norm_sum": total, "grad_absmax": max_g, "grad_tensors": count}

    # def _print_one_time_sanity(self, y_hat_col, dx_col, y_pred_data, loss_data_n, loss_data_raw):
    #     # 1) output scale sanity
    #     with torch.no_grad():
    #         self.l.info("[SANITY] ---- One-time debug ----")
    #         self.l.info(f"[SANITY] y_hat_col mean={y_hat_col.mean(0).detach().cpu().numpy()} "
    #                     f"std={y_hat_col.std(0).detach().cpu().numpy()} "
    #                     f"min={y_hat_col.min(0).values.detach().cpu().numpy()} "
    #                     f"max={y_hat_col.max(0).values.detach().cpu().numpy()}")
    #         self.l.info(f"[SANITY] dx_col mean={dx_col.mean(0).detach().cpu().numpy()} "
    #                     f"std={dx_col.std(0).detach().cpu().numpy()} "
    #                     f"mse_comp={torch.mean(dx_col * dx_col, dim=0).detach().cpu().numpy()}")
    #
    #         self.l.info(f"[SANITY] y_data mean={self.y_data.mean(0).detach().cpu().numpy()} "
    #                     f"std={self.y_data.std(0).detach().cpu().numpy()} "
    #                     f"min={self.y_data.min(0).values.detach().cpu().numpy()} "
    #                     f"max={self.y_data.max(0).values.detach().cpu().numpy()}")
    #
    #         self.l.info(f"[SANITY] y_pred_data mean={y_pred_data.mean(0).detach().cpu().numpy()} "
    #                     f"std={y_pred_data.std(0).detach().cpu().numpy()} "
    #                     f"min={y_pred_data.min(0).values.detach().cpu().numpy()} "
    #                     f"max={y_pred_data.max(0).values.detach().cpu().numpy()}")
    #
    #         self.l.info(f"[SANITY] data_norm={float(loss_data_n.item()):.3e} "
    #                     f"data_raw={float(loss_data_raw.item()):.3e} "
    #                     f"lambda_data={self.lambda_data}")
    #
    #         # probe
    #         y_probe, dx_probe, dxn = self._debug_probe()
    #         self.l.info(
    #             f"[SANITY] u_probe={tuple(self.u_probe.tolist())} y_hat={y_probe} dx_probe={dx_probe} ||dx||={dxn:.3e}")
    #
    #         # last layer stats
    #         stats = self._debug_param_and_grad_stats()
    #         if stats:
    #             self.l.info(f"[SANITY] last layer stats: {stats}")
    #         self.l.info("[SANITY] -----------------------")
    #
    # def _debug_check_ranges_and_nans(self):
    #     # Check u ranges
    #     def _rng(t: torch.Tensor):
    #         return t.min(dim=0).values.detach().cpu().numpy(), t.max(dim=0).values.detach().cpu().numpy()
    #
    #     ucol_min, ucol_max = _rng(self.u_col)
    #     uval_min, uval_max = _rng(self.u_val)
    #     udat_min, udat_max = _rng(self.u_data)
    #
    #     self.l.info(f"[CHECK] u_col min={ucol_min} max={ucol_max}")
    #     self.l.info(f"[CHECK] u_val min={uval_min} max={uval_max}")
    #     self.l.info(f"[CHECK] u_data min={udat_min} max={udat_max}")
    #     self.l.info(f"[CHECK] expected u_min={self.u_min_train} u_max={self.u_max_train}")
    #
    #     # Check y ranges
    #     y_min = self.y_data.min(dim=0).values.detach().cpu().numpy()
    #     y_max = self.y_data.max(dim=0).values.detach().cpu().numpy()
    #     self.l.info(f"[CHECK] y_data min={y_min} max={y_max}")
    #     self.l.info(f"[CHECK] y_mu={self.y_mu.detach().cpu().numpy().reshape(-1)}")
    #     self.l.info(f"[CHECK] y_std={self.y_std.detach().cpu().numpy().reshape(-1)}")
    #
    #     # NaN/inf checks
    #     def _finite(name, t):
    #         ok = torch.isfinite(t).all().item()
    #         self.l.info(f"[CHECK] {name} finite={ok}")
    #         if not ok:
    #             bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:10].detach().cpu().numpy()
    #             self.l.info(f"[CHECK] {name} first bad idx: {bad}")
    #
    #     _finite("u_col", self.u_col)
    #     _finite("u_val", self.u_val)
    #     _finite("u_data", self.u_data)
    #     _finite("y_data", self.y_data)
    #     _finite("res_dx_data", self.res_dx_data)
    #
    # @torch.no_grad()
    # def _debug_probe(self):
    #     self.net.eval()
    #     y_probe = self.net(self._u_probe_t)
    #     dx_probe = self.physics_f(y_probe, self._u_probe_t)
    #     return (
    #         y_probe.detach().cpu().numpy().reshape(-1),
    #         dx_probe.detach().cpu().numpy().reshape(-1),
    #         float(torch.norm(dx_probe).item()),
    #     )
    #
    # def _debug_param_and_grad_stats(self):
    #     # Params: last layer is usually where scale comes from
    #     last = None
    #     for m in self.net.modules():
    #         if isinstance(m, torch.nn.Linear):
    #             last = m
    #     if last is None:
    #         return {}
    #
    #     with torch.no_grad():
    #         w = last.weight
    #         b = last.bias
    #         stats = {
    #             "last_w_norm": float(torch.norm(w).item()),
    #             "last_b_norm": float(torch.norm(b).item()) if b is not None else 0.0,
    #             "last_w_absmax": float(w.abs().max().item()),
    #             "last_b_absmax": float(b.abs().max().item()) if b is not None else 0.0,
    #         }
    #     return stats
