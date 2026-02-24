import torch
from torchdiffeq import odeint

def simulate_full_range(
                    f,
                    y0,
                    u0,
                    simulation_range,
                    num_inner_steps,
                    dt_step,
                    device="cpu",
                    t0=0):
    """
    This is the core utility function for simulating the plant behavior (regarding the evolution of the states)

    Uses RK4 to compute the ground truth solution from t=0 to t=T, but this T can be any T you want
    """

    t_rk4=torch.linspace(t0,simulation_range,num_inner_steps).to(device)

    if not torch.is_tensor(y0):
        y0=torch.tensor(y0,dtype=torch.float32).to(device)
    else:
        y0=y0.clone().detach().to(device)

    if not torch.is_tensor(u0):
        u0=torch.tensor(u0,dtype=torch.float32).to(device)
    else:
        u0=u0.clone().detach().to(device)

    with torch.no_grad():
        Y_rk4=odeint(lambda t, y: f(y.view(1,-1),u0.view(1,-1)),
                     y0.view(1,-1),
                     t_rk4,
                     method="rk4",
                     options=dict(step_size=dt_step)
                     ).squeeze(1)
    return t_rk4, Y_rk4