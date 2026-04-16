import numpy as np
from Application.model_analysis_application import make_model
from Solvers.solve_glc_ode_equilibrium import solve_equilibrium_ipopt
from matplotlib import pyplot as plt

u1=0.5
u2_grid=np.linspace(0.10,1.001,20)
RES_TOL_DX=1e-6
RES_TOL_G=1e-6
y_guess_rig = [3679.08033973,
           289.73390193,
           3167.56224658,
           1041.96126532,
           50.46858403,
           759.52720527,
           249.84447542]
z_guess_rig = [8.75897957e+06,
           8.42155186e+06,
           2.17230613e+01,
           2.17230613e+01]
y_guess_sur = np.array([3285.42, 300.822, 6910.91], dtype=float)
results_all={}

model_rig = make_model("rigorous", BSW=0.20, GOR=0.05, PI=3.0e-6)
y_guess = np.array(y_guess_rig, dtype=float).reshape(-1)
if y_guess.size != model_rig["nx"]:
    raise ValueError(f"y_guess_init has size {y_guess.size}, but model nx={model_rig["nx"]}")

y_guess_init=y_guess_rig
z_guess_init=z_guess_rig


is_dae=bool(model_rig["is_dae"])

Z_NAMES=model_rig["Z_NAMES"]
n_out=len(Z_NAMES)

Nu2=len(u2_grid)

OUT = {name: np.full((Nu2), np.nan, dtype=float) for name in Z_NAMES}
RES_DX= np.full((Nu2), np.nan, dtype=float)
RES_G=np.full((Nu2), np.nan, dtype=float) if is_dae else None
STABLE= np.full((Nu2), np.nan, dtype=float)

SUCCESS = np.zeros((Nu2), dtype=bool)

j_iter = range(Nu2 - 1, -1, -1)

prev_row_rightmost_y = None
prev_row_rightmost_z = None  # only used if is_dae

for i in range(Nu2):
    u2 = u2_grid[i]
    print("\n----------------------------------")
    print(f"u2={u2}")
    try:
        if is_dae:
            y_star, z_star, dx_star, g_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
                model=model_rig,
                u_val=[u1, u2],
                y_guess=y_guess_rig,
                z_guess=z_guess_rig
            )
        else:
            y_star, dx_star, out_star, eig, stable, stats = solve_equilibrium_ipopt(
            model=model_rig,
            u_val=[u1, u2],
            y_guess=y_guess
            )

        print("IPOPT success:", bool(stats.get("success", False)))
        print("IPOPT status :", stats.get("return_status", ""))

        dx_np = np.array(dx_star, dtype=float).reshape(-1)
        res_dx=float(np.linalg.norm(dx_np))
        RES_DX[i]=res_dx

        if is_dae:
            g_np=np.array(g_star, dtype=float).reshape(-1)
            res_g=float(np.linalg.norm(g_np))
            RES_G[i]=res_g

            if (res_dx > RES_TOL_DX) or (res_g > RES_TOL_G):
                print(f"High residuals: ||dx||={res_dx:.3e}, ||g||={res_g:.3e}")
                continue
        else:
            if res_dx > RES_TOL_DX:
                print(f"High residual: ||dx||={res_dx:.3e}")
                continue

        out_np = np.array(out_star, dtype=float).reshape(-1)
        if out_np.size != n_out:
            print(f"WARNING: out size {out_np.size} != Z_NAMES {n_out}. Not storing OUT for this point.")
        else:
            for k, name in enumerate(Z_NAMES):
                OUT[name][i] = out_np[k]

        STABLE[i] = 1.0 if stable else 0.0
        SUCCESS[i] = True

        if stable:
            y_guess=np.array(y_star,dtype=float).reshape(-1)
            if is_dae:
                z_guess=np.array(z_star,dtype=float).reshape(-1)

        print("Accepted. y*:",y_star,"||dx||:", res_dx, "stable:", stable)
    except Exception as e:
        print("Exception:", repr(e))

print(OUT['w_o_out'])



fig = plt.figure(figsize=(10, 6))
plt.plot(u2_grid,OUT['w_o_out'],color='black')
plt.title('Output vs u2_grid', fontsize=14)

# Axis labels
plt.xlabel('u2_grid', fontsize=12)
plt.ylabel('w_o_out', fontsize=12)

plt.tight_layout()
plt.show()


    # return {
    #     "OUT": OUT,
    #     "RES_DX": RES_DX,
    #     "RES_G":RES_G,
    #     "STABLE": STABLE,
    #     "SUCCESS": SUCCESS,
    #     "u1_grid": u1_grid,
    #     "u2_grid": u2_grid,
    #     "Z_NAMES": Z_NAMES,
    #     "is_dae": is_dae,
    #     "P_max": P_max
    # }
#
#
# results_all["rigorous"]