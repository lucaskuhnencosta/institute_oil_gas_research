import numpy as np
import matplotlib.pyplot as plt
from Utilities.simulation_utils import simulate_full_range
# from OLGA_model.glc_OLGA import glc_rto_f_olga
from Rigorous_ODE_Model.glc_states import glc_rto_f
from Rigorous_ODE_Model.glc_algebraic import glc_f_alg
import torch



state_size = 3
control_size = 2

total_time = 15000
T_step=10
y0=[3919.7688, 437.16663, 7956.1206]
u0 = np.array([0.3, 0.60])
num_control_steps=int(total_time/T_step)
pytorch_output_dt = 1    # Save one output point every second for a smooth plot
pytorch_internal_dt = 0.5  # Very small internal step for high accuracy


t_truth, y_truth = simulate_full_range(f=glc_rto_f,
                                       y0=y0,
                                       u0=u0,
                                       simulation_range=total_time,
                                       num_inner_steps=int(total_time/pytorch_output_dt)+1,
                                       dt_step=pytorch_internal_dt)

y = y_truth
finite = torch.isfinite(y).all(dim=1)

if (~finite).any():
    k = torch.where(~finite)[0][0].item()
    print("First non-finite at index:", k, "time:", float(t_truth[k]))
    print("State there:", y[k])

    # inspect last finite state
    y_prev = y[k-1]
    print("Last finite at index:", k-1, "time:", float(t_truth[k-1]))
    print("State last finite:", y_prev)

    u_t = torch.tensor(u0, device=y_prev.device, dtype=y_prev.dtype)
    out = glc_f_alg(y_prev.view(1, 3), u_t.view(1, 2))  # shape (1,6) if you keep dim=1
    P_bh_bar, w_L_out, w_G_inj, w_G_in, w_G_out, P_tb_t = out[0]

    print(f"P_bh_bar:{P_bh_bar}")
    print(f"w_L_out:{w_L_out}")
    print(f"w_G_inj:{w_G_inj}")
    print(f"w_G_in:{w_G_in}")
    print(f"w_G_out:{w_G_out}")
    print(f"P_tb_t:{P_tb_t}")
else:
    print("All finite up to final time.")
    print("Final state:", y[-1].detach().cpu().numpy())
    y_prev = y[-1]
    u_t = torch.tensor(u0, device=y_prev.device, dtype=y_prev.dtype)
    out = glc_f_alg(y_prev.view(1, 3), u_t.view(1, 2))  # shape (1,6) if you keep dim=1
    P_bh_bar, w_L_out, w_G_inj, w_G_in, w_G_out, P_tb_t = out[0]
    print(f"P_bh_bar:{P_bh_bar}")
    print(f"w_L_out:{w_L_out}")
    print(f"w_G_inj:{w_G_inj}")
    print(f"w_G_in:{w_G_in}")
    print(f"w_G_out:{w_G_out}")
    print(f"P_tb_t:{P_tb_t}")
    # print(f"alpha:{alpha}")

# low_res_integrator = build_casadi_rk4_integrator(casadi_glc_ode,
#                                                  state_size=state_size,
#                                                  control_size=control_size,
#                                                  dtc=T_step,
#                                                  number_finite_elements=4)

t_low_res = np.linspace(0, total_time, num_control_steps + 1)
y_low_res = [np.array(y0)]
current_y_low = np.array(y0)

# for _ in range(num_control_steps):
#     res = low_res_integrator(x0=current_y_low, p=u0)
#     current_y_low = res['xf'].full().flatten()
#     y_low_res.append(current_y_low)
# y_low_res = np.array(y_low_res)

# print(f'The shape of the low resolution vector is:',y_low_res.shape)
print(f'The shape of the ground truth vector is:',y_truth.shape)


indices = np.round(np.linspace(0, len(y_truth) - 1, len(y_low_res))).astype(int)
print(indices)

y_truth_at_low_res_steps = y_truth[indices].detach().numpy()
# print(y_truth_at_low_res_steps)
# print(y_low_res)
mse = np.mean((y_low_res - y_truth_at_low_res_steps)**2)
print(f"\nMSE between low-res CasADi and high-res PyTorch: {mse:.6f}")

usa_flag_red = '#B22234'
usa_flag_blue = '#0A3161'

fig, axes = plt.subplots(state_size, 1, figsize=(12, 10), sharex=True)

t_truth_np, y_truth_np = t_truth.numpy(), y_truth.detach().numpy()
state_labels = ['Gas in Annulus [kg]', 'Gas in Tubing [kg]', 'Liquid in Tubing [kg]']
colors=[usa_flag_blue,usa_flag_red,usa_flag_red]

for i, ax in enumerate(axes):
    ax.plot(t_truth_np, y_truth_np[:, i], label='Ultra-High-Res PyTorch Truth', color=colors[i])
    # ax.plot(t_low_res, y_low_res[:, i], 'o', label=f'CasADi RK4 (h={T_step}s)', color=colors[i])
    ax.set_ylabel(state_labels[i], fontsize=12)
    if i == 0:
        ax.set_title("Open-Loop Simulation of Gas-Lift Well", fontsize=14)
        ax.legend(loc='upper right', framealpha=1, fontsize=12)
axes[-1].set_xlabel('Time [s]', fontsize=12)
plt.xlim(0, total_time)

plt.tight_layout()
plt.savefig('well_simulation_comparison_final.pdf', dpi=300, bbox_inches='tight',facecolor='0.75')
plt.show()