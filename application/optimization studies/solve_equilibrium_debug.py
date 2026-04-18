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
K_gs = wells[well]["K_gs_sur"]
K_inj = wells[well]["K_inj_sur"]
K_pr = wells[well]["K_pr_sur"]
y_guess_rig = wells[well]["y_guess_rig"]
z_guess_rig = wells[well]["z_guess_rig"]
y_guess_sur = wells[well]["y_guess_sur"]

u = [0.57724538, 0.63318548]


well_func = make_glc_well_rigorous(BSW=BSW,
                                GOR=GOR,
                                PI=PI,
                                K_gs=K_gs,
                                K_inj=K_inj,
                                K_pr=K_pr)

model_rigorous = build_steady_state_model(
f_func=well_func,
state_size=7,
control_size=2,
alg_size=4,
name="glc_ss_rigorous",
out_name=Z_NAMES
)


y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
model=model_rigorous,
u_val=u,
y_guess=y_guess_rig,
z_guess=z_guess_rig,
)

# Pretty-print the OUT vector by name (uses model["Z_names"])
print("\n--- out* (named) ---")
print_z_grouped(out_star, model_rigorous["Z_NAMES"])
