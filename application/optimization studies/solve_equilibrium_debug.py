import numpy as np

from simulators.black_box_simulator.black_box_model import make_glc_well_rigorous
from simulators.surrogate_simulator.surrogate_model_casadi import make_glc_well_surrogate
from utilities.block_builders import build_steady_state_model
from utilities.solver_utilities import print_z_grouped
from simulators.Z_NAMES import Z_NAMES
from configuration.wells import get_wells
from solvers.steady_state_solver import solve_equilibrium_ipopt


wells = get_wells()
well="P1"
BSW = wells[well]["BSW"]
GOR = wells[well]["GOR"]
PI = wells[well]["PI"]
K_gs = wells[well]["K_gs"]
K_inj = wells[well]["K_inj"]
K_pr = wells[well]["K_pr"]
y_guess_rig = wells[well]["y_guess_rig"]
z_guess_rig = wells[well]["z_guess_rig"]
y_guess_sur = wells[well]["y_guess_sur"]
sim_kind="rigorous"
#
# well_func = make_glc_well_rigorous(BSW=BSW,

#
# model_surrogate = build_steady_state_model(
# f_func=well_func,
# state_size=7,
# control_size=2,
# name="glc_ss_rigorous",
# out_name=Z_NAMES
# )


from application.simulation_engine import make_model

model=make_model(sim_kind="rigorous",
                        BSW=BSW,
                        GOR=GOR,
                        PI=PI,
                        K_gs=K_gs,
                        K_inj=K_inj,
                        K_pr=K_pr)


u_P1 = np.array([0.20699727289545305, 0.5249146997717429])
u_P2=[0.6894070850915028, 0.24133944946614966]
y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
model=model,
u_val=u_P1,
y_guess=y_guess_rig,
z_guess=z_guess_rig
)

# Pretty-print the OUT vector by name (uses model["Z_names"])
print("\n--- out* (named) ---")
print_z_grouped(out_star, model["Z_NAMES"])
