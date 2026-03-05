import torch
import wandb
import numpy as np

from Surrogate_ODE_Model.glc_surrogate_torch import glc_surrogate_dx_torch
from Surrogate_ODE_Model.glc_check_feasibility import glc_check_feasibility
from OLD.networks import PINN
from Training.base_trainer import Trainer

from contextlib import nullcontext

class PINCWellTrainer(Trainer):
    def __init__(self,
                net: PINN,
                N_data: int=1000, # Pairs (y1,y2,y3,u1,u2), (dy1,dy2,dy3)
                N_col: int=10000, # Points where derivative will be calculated (y1,y2,y3,u1,u2)
                N_val: int=100,
                adam_epochs: int=1000,
                lbfgs_epochs: int=2000,
                lr: float=7e-3,
                w_phys: float =10.0,
                mse_f_scale_factors: list=[1.0e3,1.0e3,1.0e2],
                y_min: list=[3030.0,220.0,6345.0],
                y_max: list=[4800.0,1095.0,11995.0],
                u_min: list=[0.05,0.10],
                u_max: list=[1.0,1.0],
                mixed_precision=True,
                device=None,
                wandb_project="PINC_GasLift_SS",
                wandb_group=None,
                random_seed=333333):
        self.N_y=3 # Number of states
        self.N_u=2 # Number of controls

        self.K1=adam_epochs
        self.K2=lbfgs_epochs

        super().__init__(net=net,
                         epochs=adam_epochs+lbfgs_epochs,
                         lr=lr,
                         optimizer='Adam',
                         loss_func='MSELoss',
                         lr_scheduler=None,
                         mixed_precision=mixed_precision,
                         device=device,
                         wandb_project=wandb_project,
                         wandb_group=wandb_group,
                         random_seed=random_seed)

        self.N_data=N_data
        self.N_col=N_col
        self.N_val=N_val
        self.w_phys=w_phys

        #Store domain bounds for data generation
        self.y_min_train=np.array(y_min)
        self.y_max_train=np.array(y_max)
        self.u_min_train=np.array(u_min)
        self.u_max_train=np.array(u_max)

        # bounds for scaling (torch)
        self.y_min_t = torch.tensor(y_min, dtype=torch.float32, device=self.device)
        self.y_max_t = torch.tensor(y_max, dtype=torch.float32, device=self.device)
        self.u_min_t = torch.tensor(u_min, dtype=torch.float32, device=self.device)
        self.u_max_t = torch.tensor(u_max, dtype=torch.float32, device=self.device)
        self.y_range_t = (self.y_max_t - self.y_min_t)
        self.u_range_t = (self.u_max_t - self.u_min_t)

        self.mse_f_scale = torch.tensor(mse_f_scale_factors, device=self.device)

        self.physics_f=glc_surrogate_dx_torch
        self.gt_f=glc_surrogate_dx_torch # For now, it is the same

        self._add_to_wandb_config({
            'N_data':self.N_data,
            'N_col':self.N_col,
            'N_val':self.N_val,
            'K1_adam': self.K1,
            'K2_lbfgs': self.K2,
            'mse_f_scale_factors': mse_f_scale_factors,
            'y_min':y_min,
            'y_max':y_max,
            'u_min':u_min,
            'u_max':u_max,
        })

    def _scale_y(self, y):
        return 2.0 * (y - self.y_min_t) / self.y_range_t - 1.0
    def _scale_u(self, u):
        return 2.0 * (u - self.u_min_t) / self.u_range_t - 1.0
    def _unscale_dy(self, dy_scaled):
        # dy = (y_range/2) * d(y_scaled)/dt
        return 0.5 * self.y_range_t * dy_scaled
    def _scale_dy(self, dy):
        # dy_scaled = 2*dy/y_range
        return 2.0 * dy / self.y_range_t

    def prepare_data(self):
        self.l.info("Preparing PINN data for Gas-Lift Well...")

        ############################################################
        # 1) Supervised "data" points
        self.l.info(f"Generating {self.N_ic} initial condition points...")
        np.random.seed(333333)

        y_np=np.random.uniform(self.y_min_train,self.y_max_train,(self.N_data,self.N_y))
        u_np=np.random.uniform(self.u_min_train,self.u_max_train,(self.N_data,self.N_u))

        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        u = torch.tensor(u_np, dtype=torch.float32, device=self.device)

        feasible=glc_check_feasibility(y,u)
        y=y[feasible]
        u=u[feasible]

        self.N_new_data=len(y)
        self.l.info(f"  Removed {self.N_new_data - self.N_data} infeasible IC points. New N_ic = {self.N_new_data}")

        # label with "ground truth" dynamics
        with torch.no_grad():
            dy = self.gt_f(y, u)

        # extra safety: remove any non-finite labels
        finite = torch.isfinite(dy).all(dim=1)
        y = y[finite]
        u = u[finite]
        dy = dy[finite]

        # store scaled versions for training
        self.y_data = y
        self.u_data = u
        self.y_data_s = self._scale_y(y)
        self.u_data_s = self._scale_u(u)
        self.dy_data_s = self._scale_dy(dy)  # train target in scaled derivative space

        self.data_y = (self.y_data_s, self.u_data_s, self.dy_data_s)


        ############################################################
        # 2) Collocation (physics) points
        self.l.info(f"Generating {self.N_col} collocation points...")

        y_col_np=np.random.uniform(self.y_min_train,self.y_max_train,(self.N_col,self.N_y))
        u_col_np=np.random.uniform(self.u_min_train,self.u_max_train,(self.N_col,self.N_u))

        y_col=torch.tensor(y_col_np,dtype=torch.float32,device=self.device)
        u_col=torch.tensor(u_col_np,dtype=torch.float32,device=self.device)

        feasible_col = glc_check_feasibility(y_col, u_col)
        y_col = y_col[feasible_col]
        u_col = u_col[feasible_col]

        self.N_new_col = len(y_col)
        self.l.info(f"  Removed {self.N_new_col - self.N_col} infeasible IC points. New N_ic = {self.N_new_col}")

        self.y_col = y_col
        self.u_col = u_col
        self.y_col_s = self._scale_y(y_col)
        self.u_col_s = self._scale_u(u_col)

        self.data_f = (self.y_col_s, self.u_col_s)

        ############################################################
        # 3) Validation (dy match) points

        self.l.info(f"Generating {self.N_val} validation points...")
        np.random.seed(1)

        yv_np = np.random.uniform(self.y_min_train, self.y_max_train, (self.N_val, self.N_y))
        uv_np = np.random.uniform(self.u_min_train, self.u_max_train, (self.N_val, self.N_u))

        yv = torch.tensor(yv_np, dtype=torch.float32, device=self.device)
        uv = torch.tensor(uv_np, dtype=torch.float32, device=self.device)

        feas_v = glc_check_feasibility(yv, uv)
        yv = yv[feas_v]
        uv = uv[feas_v]

        with torch.no_grad():
            dyv = self.gt_f(yv, uv)
        finite_v = torch.isfinite(dyv).all(dim=1)
        yv = yv[finite_v]
        uv = uv[finite_v]
        dyv = dyv[finite_v]

        self.val_data = (self._scale_y(yv), self._scale_u(uv), self._scale_dy(dyv))

        self.l.info("Data preparation complete!")

        self.print_shapes()

    def print_shapes(self):
        """
        Logs the shapes of all prepared data tensors for quick debugging.
        """
        self.l.info("--- Data Tensor Shapes ---")

        # --- Initial Condition Data ---
        y_data_s, u_data_s, dy_data_s = self.data_y
        self.l.info(f"Initial Condition (IC) Data (N_ic = {y_data_s.shape[0]}):")
        self.l.info(f"  y0_ic (input):     {y_data_s.shape}")
        self.l.info(f"  u_ic (input):      {u_data_s.shape}")
        self.l.info(f"  y0_ic (target):    {dy_data_s.shape}")  # y0_ic is both input and target

        # --- Collocation (Physics) Data ---
        y_col_s, u_col_s = self.data_f
        self.l.info(f"Collocation (Physics) Data (N_col = {y_col_s.shape[0]}):")
        self.l.info(f"  y0_col (input):    {y_col_s.shape}")
        self.l.info(f"  u_col (input):     {u_col_s.shape}")

        # --- Validation Data ---
        y_val_s, u_val_s, y_target_val_s = self.val_data
        self.l.info(f"Validation Data (N_val = {y_val_s.shape[0]}):")
        self.l.info(f"  y0_val (input):    {y_val_s.shape}")
        self.l.info(f"  u_val (input):     {u_val_s.shape}")
        self.l.info(f"  y_target_val (target): {y_target_val_s.shape}")

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
    def get_physics_loss(self,y_s,u_s):
        """
        Calculates the physics loss (MSE_F).

        This involves:
        1. Calling the PINC network
        2. Calculating the derivative dy/dt w.r.t time t.
        3. Calculating the "true" derivatives f(y,u) using the physics model.
        4. Computing the scaled MSE between them.
        """

        # 1. Call the PINC network
        dy_pred_s=self.net(y_s,u_s)
        dy_pred = self._unscale_dy(dy_pred_s)

        y = 0.5 * (y_s + 1.0) * self.y_range_t + self.y_min_t
        u = 0.5 * (u_s + 1.0) * self.u_range_t + self.u_min_t

        dy_true=self.physics_f(y,u)

        # 2. Calcualte the derivative dy/dt
        # We need to compute the gradient of each output state w.r.t the time input
        dy_dt_preds_list=[]
        for i in range(self.N_y):
            # Summing rhe output for the ith state component
            # create_graph=True is essential to L-BFGS
            grad_outputs=torch.ones_like(y_pred_f[:,i],device=self.device)
            dy_i_dt=torch.autograd.grad(
                outputs=y_pred_f[:,i],
                inputs=t_f,
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]
            dy_dt_preds_list.append(dy_i_dt)

        dy_dt_preds=torch.cat(dy_dt_preds_list,dim=1)

        # 3. Calculate the "true" derivatives f(y,u)
        # We use the safeguarded physics function here
        dy_dt_true=self.physics_f(y_pred_f,u_f)

        # 4. Compute the scaled MSE
        # This normalization matches the TF code:
        y_range=(self.net.y_max-self.net.y_min)
        residual=dy_dt_preds-dy_dt_true

        # This computes the MSE for each component (m_G_an,m_G_tb,m_L_tb)
        loss_f_components=torch.mean((residual/y_range)**2,dim=0)

        # Apply the scale factors (e.g., [1e3,1e3,1e3])
        scaled_loss_f_components=self.mse_f_scale*loss_f_components

        # The final loss is the mean of the scaled components
        loss_f=torch.mean(scaled_loss_f_components)

        return loss_f,loss_f_components

    def train_pass(self):
        """
        Performs a single training pass (one epoch for Adam, or one
        "one-shot" call for L-BFGS).
        """
        self.net.train()

        (t_ic,y0_ic,u_ic) = self.data_y
        (t_f,y0_f,u_f) = self.data_f

        if self.optimizer == 'LBFGS':
            def closure():
                self._optim.zero_grad()

                # 1. IC Loss *MSE_y)
                y_pred_ic=self.net(t_ic,y0_ic,u_ic)
                # y0_ic is the target
                y_range=(self.net.y_max-self.net.y_min)
                loss_y_components=torch.mean(((y_pred_ic-y0_ic)/y_range)**2,dim=0)
                loss_y=torch.mean(loss_y_components)

                # 2. Physics Loss (MSE_f)
                loss_f,loss_f_comps=self.get_physics_loss(t_f,y0_f,u_f)

                #3. Total Loss
                loss=loss_y+10.0*loss_f

                loss.backward()

                self._current_losses = {
                    'total': loss.item(),
                    'initial_condition': loss_y.item(),
                    'physics': loss_f.item(),
                    'physics_comp_0': loss_f_comps[0].item(),
                    'physics_comp_1': loss_f_comps[1].item(),
                    'physics_comp_2': loss_f_comps[2].item(),
                }
                return loss
            self._optim.step(closure)
            losses=self._current_losses
        else:
            if torch.is_grad_enabled():
                self._optim.zero_grad()

            with self.autocast_if_mp():
                # 1. IC Loss (MSE_y)
                y_pred_ic=self.net(t_ic,y0_ic,u_ic)
                y_range=(self.net.y_max-self.net.y_min)
                loss_y_components=torch.mean(((y_pred_ic-y0_ic)/y_range)**2,dim=0)
                loss_y=torch.mean(loss_y_components)

                # 2. Pjhysics Loss (MSE_f)
                loss_f,loss_f_comps=self.get_physics_loss(t_f,y0_f,u_f)

                #3. Total loss
                loss=loss_y+10.0*loss_f

                if self._scaler:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    loss.backward()
                    self._optim.step()

                losses = {
                    'total': loss.item(),
                    'initial_condition': loss_y.item(),
                    'physics': loss_f.item(),
                    'physics_comp_0': loss_f_comps[0].item(),
                    'physics_comp_1': loss_f_comps[1].item(),
                    'physics_comp_2': loss_f_comps[2].item(),
                }

        return losses

    def validation_pass(self):
        """
        Performs the single-step validation, matching the TF code's
        'validate_MSE' function.
        """

        self.net.eval()

        (t_val,y0_val,u_val,y_target_val) = self.val_data

        with torch.no_grad():
            with self.autocast_if_mp():
                # 1. Get PINC's single-step
                y_pred_val=self.net(t_val,y0_val,u_val)

                # 2. Compare to RK4 found trutg targer
                y_range=(self.net.y_max-self.net.y_min)
                val_loss_components=torch.mean(((y_pred_val-y_target_val)/y_range)**2,dim=0)
                val_loss=torch.mean(val_loss_components).item()
        losses = {
            'total': val_loss,
            'val_comp_0': val_loss_components[0].item(),
            'val_comp_1': val_loss_components[1].item(),
            'val_comp_2': val_loss_components[2].item(),
        }
        return losses

    def run(self):
        """
        Custom run-loop to override the base_trainer's and match the
        "Adam -> LBFGS_Block_1 -> LBFGS_Block_2" pattern.
        """
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

    def _log_epoch_info(self,train_losses: dict, val_losses: dict):
        """
        Implements the abstract logging method from the base Trainer

        This prints a formatted summary of the training and validation losses
        for the current epoch.
        """
        # We can access all the specific keys we defined in train_pass
        train_loss = train_losses['total']
        loss_ic = train_losses['initial_condition']
        loss_phys = train_losses['physics']

        # And the keys from validation_pass
        val_loss = val_losses['total']

        self.l.info(
            f"Epoch {self._e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"(Train IC: {loss_ic:.8f}, "
            f"Train Physics: {loss_phys:.8f})"
        )
