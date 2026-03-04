import torch
from torch.utils.data import TensorDataset, DataLoader
from Training.base_trainer import Trainer

class SupervisedWellTrainer(Trainer):
    def __init__(
        self,
        net: torch.nn.Module,
        model,
        solve_equilibrium_ipopt,
        z_names,
        n_train=5000,
        n_val=1000,
        batch_size=256,
        u_min=(0.05, 0.20),
        u_max=(1.0, 1.0),
        y_guess=(50.0, 500.0, 500.0),
        require_stable=False,
        # Trainer base params
        epochs=200,
        lr=1e-3,
        optimizer="Adam",
        loss_func="MSELoss",
        mixed_precision=True,
        device=None,
        wandb_project=None,
        wandb_group=None,
        random_seed=42,
    ):
        super().__init__(
            net=net,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            loss_func=loss_func,
            mixed_precision=mixed_precision,
            device=device,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            random_seed=random_seed,
        )
        self.model = model
        self.solve_equilibrium_ipopt = solve_equilibrium_ipopt
        self.z_names = z_names

        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size

        self.u_min = u_min
        self.u_max = u_max
        self.y_guess = y_guess
        self.require_stable = require_stable

        # filled in prepare_data
        self.u_mean = None
        self.u_std = None
        self.y_mean = None
        self.y_std = None

    def prepare_data(self):
        # 1) generate raw numpy data
        U_train, Y_train = generate_ss_dataset(
            model=self.model,
            solve_equilibrium_ipopt=self.solve_equilibrium_ipopt,
            z_names=self.z_names,
            n_samples=self.n_train,
            u_min=self.u_min,
            u_max=self.u_max,
            y_guess=self.y_guess,
            seed=self.random_seed,
            require_stable=self.require_stable,
        )
        U_val, Y_val = generate_ss_dataset(
            model=self.model,
            solve_equilibrium_ipopt=self.solve_equilibrium_ipopt,
            z_names=self.z_names,
            n_samples=self.n_val,
            u_min=self.u_min,
            u_max=self.u_max,
            y_guess=self.y_guess,
            seed=self.random_seed + 1,
            require_stable=self.require_stable,
        )

        # 2) compute normalizers on train set
        self.u_mean = U_train.mean(axis=0, keepdims=True)
        self.u_std  = U_train.std(axis=0, keepdims=True) + 1e-8
        self.y_mean = Y_train.mean(axis=0, keepdims=True)
        self.y_std  = Y_train.std(axis=0, keepdims=True) + 1e-8

        # 3) normalize
        U_train_n = (U_train - self.u_mean) / self.u_std
        Y_train_n = (Y_train - self.y_mean) / self.y_std
        U_val_n   = (U_val   - self.u_mean) / self.u_std
        Y_val_n   = (Y_val   - self.y_mean) / self.y_std

        # 4) torch datasets/loaders
        U_train_t = torch.tensor(U_train_n, dtype=torch.float32, device=self.device)
        Y_train_t = torch.tensor(Y_train_n, dtype=torch.float32, device=self.device)
        U_val_t   = torch.tensor(U_val_n,   dtype=torch.float32, device=self.device)
        Y_val_t   = torch.tensor(Y_val_n,   dtype=torch.float32, device=self.device)

        self.train_loader = DataLoader(
            TensorDataset(U_train_t, Y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            TensorDataset(U_val_t, Y_val_t),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # keep raw (unnormalized) val too if you want extra plots later
        self._val_raw = (U_val, Y_val)

        self.l.info(f"Prepared data: train={len(self.train_loader.dataset)}, val={len(self.val_loader.dataset)}")
        self.l.info(f"Targets are [w_o_out, P_bh_bar, P_tb_b_bar] (normalized).")

    def train_pass(self):
        self.net.train()
        total_loss = 0.0
        n = 0

        for U, Y in self.train_loader:
            self._optim.zero_grad()

            with self.autocast_if_mp():
                Y_hat = self.net(U)
                loss = self._loss_func(Y_hat, Y)

            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optim)
                self._scaler.update()
            else:
                loss.backward()
                self._optim.step()

            bs = U.shape[0]
            total_loss += loss.item() * bs
            n += bs

        total_loss /= max(n, 1)
        return {"total": total_loss}

    def validation_pass(self):
        self.net.eval()
        total_loss = 0.0
        total_mae = np.zeros(3, dtype=np.float64)
        n = 0

        with torch.no_grad():
            for U, Y in self.val_loader:
                with self.autocast_if_mp():
                    Y_hat = self.net(U)
                    loss = self._loss_func(Y_hat, Y)

                bs = U.shape[0]
                total_loss += loss.item() * bs
                n += bs

                # MAE in normalized units
                mae = torch.mean(torch.abs(Y_hat - Y), dim=0).detach().cpu().numpy()
                total_mae += mae * bs

        total_loss /= max(n, 1)
        total_mae /= max(n, 1)

        return {
            "total": float(total_loss),
            "mae_w_o_out": float(total_mae[0]),
            "mae_P_bh": float(total_mae[1]),
            "mae_P_tb_b": float(total_mae[2]),
        }

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        self.l.info(
            f"Epoch {self._e} | "
            f"Train: {train_losses['total']:.6e} | "
            f"Val: {val_losses['total']:.6e} | "
            f"MAE(norm): "
            f"[w_o_out {val_losses['mae_w_o_out']:.3e}, "
            f"P_bh {val_losses['mae_P_bh']:.3e}, "
            f"P_tb_b {val_losses['mae_P_tb_b']:.3e}]"
        )

    # helper for inference in original units
    def predict_physical(self, u_np: np.ndarray) -> np.ndarray:
        """
        u_np: (N,2) in original units (0..1)
        returns y_np: (N,3) in original units:
            [w_o_out (kg/s), P_bh_bar (bar), P_tb_b_bar (bar)]
        """
        self.net.eval()
        u_n = (u_np - self.u_mean) / self.u_std
        u_t = torch.tensor(u_n, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat_n = self.net(u_t).cpu().numpy()
        y_hat = y_hat_n * self.y_std + self.y_mean
        return y_hat