from contextlib import nullcontext
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import autocast

class Trainer(ABC):
    """Generic trainer for PyTorch NNs."""
    def __init__(self,
                 net: nn.Module,
                 epochs=5,
                 lr=0.1,
                 optimizer: str = 'Adam',
                 optimizer_params: dict = None,
                 loss_func: str = 'MSELoss',
                 lr_scheduler: str = None,
                 lr_scheduler_params: dict = None,
                 mixed_precision=True,
                 device=None,
                 wandb_project=None,
                 wandb_group=None,
                 logger=None,
                 checkpoint_every=50,
                 random_seed=42
                 ) -> None:

        self._is_initialized = False

        if device is None:
            _device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(_device_str)
        else:
            # This correctly handles strings like 'cuda', 'cuda:0', or 'cpu'
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self._e = 0  # inital epoch
        self.epochs = epochs
        self.lr = lr
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.loss_func_name = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params if lr_scheduler_params is not None else {}
        self._dtype = next(self.net.parameters()).dtype
        self.mixed_precision = mixed_precision
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger
        self.checkpoint_every = checkpoint_every
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')
        self.train_loss_at_best_val=float('inf')

        self._log_to_wandb = wandb_project is not None
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.train_losses_list = []
        self.val_losses_list = []
        self.iae_list = []
        self.mae_list = []
        self._wandb_config = {}


    def setup_training(self):
        self.l.info('Setting up training')

        # Optimizer = eval(f"torch.optim.{self.optimizer}")
        Optimizer= getattr(torch.optim, self.optimizer)
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )

        if self.lr_scheduler is not None:
            self.l.info(f"using LR scheduler: {self.lr_scheduler} with params: {self.lr_scheduler_params}")
            Scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler)
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "mixed_precision": self.mixed_precision,
                "loss_func": self.loss_func_name,
                "random_seed": self.random_seed,
                "device": str(self.device),
            })

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._loss_func=getattr(nn,self.loss_func_name)()
        self.autocast_if_mp = nullcontext
        self._scaler = None
        if self.mixed_precision:
            # Check if we are actually on a CUDA device before enabling GPU tools
            if self.device.type == 'cuda':
                self.l.info("Mixed precision training is enabled on CUDA.")
                # Use the modern, device-aware autocast
                self.autocast_if_mp = autocast
                self._scaler = torch.amp.GradScaler('cuda')
                # self.autocast_if_mp = lambda: torch.amp.autocast(device_type=self.device.type)
                # self._scaler = torch.cuda.amp.GradScaler()
            else:
                self.l.info("Mixed precision is enabled, but no CUDA device is available. Proceeding with standard precision.")
        self.l.info('Preparing data')
        self.prepare_data()
        self._is_initialized = True

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project="PINN-VDP",
            entity="lucas-kuhnen-universidade-federal-de-santa-catarina",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net)
        self._id = wandb.run.id
        self.l.info(f"Wandb set up. Run ID: {self._id}")

    @abstractmethod
    def prepare_data(self):
        """
        Must populate `self.data` and `self.val_data`.
        """

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix + k] = v

        return data_to_log

    def _log_epoch_info(self, train_losses: dict, val_losses: dict):
        """
        Must be implemented by subclass to print custom epoch info.

        Example:
        self.l.info(f"Epoch {self._e}: Train Loss = {train_losses['total']:.8f}")
        """
        pass

    def _run_epoch(self):
        train_losses = self.train_pass()
        val_losses = self.validation_pass()

        self.train_losses_list.append(train_losses['total'])
        self.val_losses_list.append(val_losses['total'])

        data_to_log = {}
        for k, v in train_losses.items():
            data_to_log[f"train/{k}"] = v  # e.g., "train/total", "train/physics"

        for k, v in val_losses.items():
            data_to_log[f"val/{k}"] = v  # e.g., "val/total", "val/mae"

        # self.iae_list.append(val_losses.get('iae', 0))
        # self.mae_list.append(val_losses.get('mae', 0))

        if self._e % 10 == 0 or self._e == self.epochs - 1:
            self._log_epoch_info(train_losses, val_losses)


        val_score=val_losses['total']
        return data_to_log, val_score

    def run(self):
        if not self._is_initialized:
            self.setup_training()
        while self._e < self.epochs:
            if hasattr(self,'K1') and self._e==self.K1:
                if hasattr(self,'switch_to_lbfgs'):
                    self.l.info("Switching to L-BFGS for fine-tuning")
                    self.switch_to_lbfgs()

            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.save_checkpoint()

            if val_score < self.best_val:
                self.best_val = val_score
                self.train_loss_at_best_val=data_to_log['train/total']
                if self._log_to_wandb:
                    self.save_model(name='model_best')


            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('training finished!')

        return {
            "train_loss": self.train_losses_list,
            "val_loss": self.val_losses_list,
            "best_val_loss": self.best_val,
            "train_loss_at_best_val": self.train_loss_at_best_val,
        }

    @abstractmethod
    def train_pass(self):
        pass

    @abstractmethod
    def validation_pass(self):
        pass

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            # 'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.optimizer != 'LBFGS':
            checkpoint['optimizer_state_dict'] = self._optim.state_dict()
        else:
            self.l.info("Skipping optimizer state in checkpoint (L-BFGS).")

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir) / 'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir) / fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath