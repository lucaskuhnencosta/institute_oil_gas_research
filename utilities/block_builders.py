import casadi as ca
from networks.networks import PINN
from networks.networks import AlgNN
import torch
import casadi as ca

from configuration.wells import get_wells
from networks.networks import PINN, AlgNN


import numpy as np
import torch
import torch.nn as nn

#######################################################################
################# LOAD MODEL WEIGHTS ################################
#######################################################################

def load_model_weights(model,model_path, device="cpu"):
    """Just loads the model from model_path and put it in evaluation mode"""
    print(f"=== LOADING MODEL FROM {model_path} ===")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    return model

#######################################################################
################# EXTRACT PINN WEIGHTS ################################
#######################################################################

def extract_pinn_standard_weights(model: nn.Module) -> dict:
    model.cpu().eval()
    W, b = [], []
    for layer in model.layers:
        W.append(layer.weight.detach().numpy())
        b.append(layer.bias.detach().numpy())
    out_w = model.out.weight.detach().numpy()
    out_b = model.out.bias.detach().numpy()
    return {"hidden_layers_w": W, "hidden_layers_b": b, "output_w": out_w, "output_b": out_b}


def extract_algnn_standard_weights(model: nn.Module) -> dict:
    model.cpu().eval()
    W, b = [], []
    for layer in model.main_layers:
        W.append(layer.weight.detach().numpy())
        b.append(layer.bias.detach().numpy())
    out_w = model.output_layer.weight.detach().numpy()
    out_b = model.output_layer.bias.detach().numpy()
    return {"hidden_layers_w": W, "hidden_layers_b": b, "output_w": out_w, "output_b": out_b}

#######################################################################
################# CASADI BUILDERS ################################
#######################################################################

def build_casadi_pinn_function_standard(
    weights_dict: dict,
    y_min: list,
    y_max: list,
    u_min: list,
    u_max: list,
    activation=ca.tanh
):

    """
    This is the final implementation of this function, which takes a Pytorch PINN and rebuilds it as a pure CasADi function
    It is essentially a symbolic copy of:

    y_hat=net(u)

    y_min and y_max are the physical minimum and maximum values of the state outputs
    They are used to convert the network's normalized output into physical units

    u_min and u_max are the physical bounds of the inputs

    activation is logically tha tanh
    """
    # ----------------------------
    # 1. Conver min/max to CasADi dense constants
    # ----------------------------
    y_min_ca = ca.DM(y_min); y_max_ca = ca.DM(y_max)
    u_min_ca = ca.DM(u_min); u_max_ca = ca.DM(u_max)

    # ----------------------------
    # 2. Compute ranges
    # ----------------------------
    y_range = y_max_ca - y_min_ca
    u_range = u_max_ca - u_min_ca

    # ----------------------------
    # 3. Infer dimension of controls
    # ----------------------------
    n_u = len(u_min)

    # ----------------------------
    # 4. Creates symbolic control u
    # ----------------------------
    u_sym = ca.MX.sym("u", n_u)

    # ----------------------------
    # 5. Input scaling
    # ----------------------------
    # u scaling (same as PINN._scale_u)
    u_scaled = 2 * (u_sym - u_min_ca) / (u_range + 1e-12) - 1
    a = u_scaled

    # ----------------------------
    # 6. Reconstructing each hidden layer one by one. Convert weights in CasADi constants
    # ----------------------------
    # hidden layers
    for W_np, b_np in zip(weights_dict["hidden_layers_w"], weights_dict["hidden_layers_b"]):
        W = ca.DM(W_np)
        b = ca.DM(b_np).reshape((-1, 1))
        a = activation(W @ a + b)

    # ----------------------------
    # 7. Output layer
    # ----------------------------
    out_w = ca.DM(weights_dict["output_w"])
    out_b = ca.DM(weights_dict["output_b"]).reshape((-1, 1))
    y_norm = out_w @ a + out_b

    # ----------------------------
    # 8. Output squashing and descalating
    # ----------------------------
    # y de-scale (same as PINN._descale_y)
    y_squash = ca.tanh(y_norm)
    y_hat = 0.5 * (y_squash + 1.0) * y_range + y_min_ca

    return ca.Function("F_pinn_u2y", [u_sym], [y_hat], ["u"], ["y"])


def build_casadi_algnn_function_standard(
    weights_dict: dict,
    y_min: list, y_max: list,
    u_min: list, u_max: list,
    z_min: list, z_max: list,
    BSW: float,
    GOR: float,
    activation=ca.tanh
):
    """
    This is the final implementation of this function

    The NN learns only:
    Z_hat=[P_bh_bar, P_tb_b_bar, w_G_inj,w_res]

    Then the remaining outputs are reconstructed analytically:
    w_L_res, w_G_res, w_w_out, w_o_out

    Final output ordering:

    [P_bh_bar,P_tb_b_bar,w_G_inj,w_res,w_L_res,w_G_res,w_w_out,w_o_out]
    """
    # ----------------------------
    # 1. Constants as CasADi DM
    # ---------------------------
    y_min_ca = ca.DM(y_min); y_max_ca = ca.DM(y_max)
    u_min_ca = ca.DM(u_min); u_max_ca = ca.DM(u_max)
    z_min_ca = ca.DM(z_min); z_max_ca = ca.DM(z_max)

    y_range = y_max_ca - y_min_ca
    u_range = u_max_ca - u_min_ca
    z_range = z_max_ca - z_min_ca

    n_y = len(y_min)
    n_u = len(u_min)

    # ----------------------------
    # 2. Symbolic inputs
    # ----------------------------
    y_sym = ca.MX.sym("y", n_y)
    u_sym = ca.MX.sym("u", n_u)

    # ----------------------------
    # 3. Input scaling
    # ----------------------------
    y_scaled = 2 * (y_sym - y_min_ca) / (y_range + 1e-12) - 1
    u_scaled = 2 * (u_sym - u_min_ca) / (u_range + 1e-12) - 1
    a = ca.vertcat(y_scaled, u_scaled)

    # ----------------------------
    # 4. Hidden layers
    # ----------------------------
    for W_np, b_np in zip(weights_dict["hidden_layers_w"], weights_dict["hidden_layers_b"]):
        W = ca.DM(W_np)
        b = ca.DM(b_np).reshape((-1, 1))
        a = activation(W @ a + b)

    # ----------------------------
    # 5. Output layer
    # ----------------------------
    out_w = ca.DM(weights_dict["output_w"])
    out_b = ca.DM(weights_dict["output_b"]).reshape((-1, 1))
    z_norm = out_w @ a + out_b

    # ----------------------------
    # 6. Linear denormalization
    # ----------------------------
    z_hat = 0.5 * (z_norm + 1.0) * z_range + z_min_ca

    # ----------------------------
    # 7. Learned outputs
    # ----------------------------
    P_bh_bar = z_hat[0]
    P_tb_b_bar = z_hat[1]
    w_G_inj=z_hat[2]
    w_res=z_hat[3]

    alpha_G_tb_b = GOR / (GOR + 1)
    w_L_res = (1 - alpha_G_tb_b) * w_res
    w_G_res = alpha_G_tb_b * w_res

    w_w_res = BSW * w_L_res
    w_o_res = (1.0 - BSW) * w_L_res

    w_w_out = w_w_res
    w_o_out = w_o_res

    z_full=ca.vertcat(
        P_bh_bar,
        P_tb_b_bar,
        w_G_inj,
        w_res,
        w_L_res,
        w_G_res,
        w_w_out,
        w_o_out
    )

    return ca.Function("F_algnn_yu2z",
                       [y_sym, u_sym],
                       [z_full],
                       ["y", "u"],
                       ["z"])

def build_casadi_surrogate_u2z(
    pinn_weights: dict,
    algnn_weights: dict,
    pinn_y_min: list, pinn_y_max: list,
    pinn_u_min: list, pinn_u_max: list,
    alg_y_min: list, alg_y_max: list,
    alg_u_min: list, alg_u_max: list,
    alg_z_min: list, alg_z_max: list,
    BSW: float,
    GOR: float,
):
    F_pinn = build_casadi_pinn_function_standard(
        weights_dict=pinn_weights,
        y_min=pinn_y_min, y_max=pinn_y_max,
        u_min=pinn_u_min, u_max=pinn_u_max
    )
    F_alg = build_casadi_algnn_function_standard(
        weights_dict=algnn_weights,
        y_min=alg_y_min, y_max=alg_y_max,
        u_min=alg_u_min, u_max=alg_u_max,
        z_min=alg_z_min, z_max=alg_z_max,
        BSW=BSW,
        GOR=GOR
    )

    u = ca.MX.sym("u", len(pinn_u_min))

    y = F_pinn(u=u)["y"]              # (3x1)
    z = F_alg(y=y, u=u)["z"]          # (8x1): [p_bh, p_tb_b,w_G_inj,w_res,w_L_res,w_G_res,w_w_out,w_o_out] depending on your z_min/z_max order

    return ca.Function("F_u2z", [u], [z], ["u"], ["z"])


def get_model_paths(well_name: str, base_dir="../../well_models"):
    """
    Returns paths for PINN and AlgNN models based on well name.

    Example:
        P1 -> well_1
        P2 -> well_2
    """
    # Extract number from "P1" -> "1"
    well_id = well_name.replace("P", "")

    well_folder = Path(base_dir) / f"well_{well_id}"

    pinn_path = well_folder / "PINN" / f"PINN_{well_name}.pth"
    algnn_path = well_folder / "Alg" / f"AlgNN_{well_name}.pth"

    return pinn_path, algnn_path



def build_casadi_surrogate_u2z_for_well(
    well_name: str,
    pinn_hidden_units=(64, 64, 64),
    alg_hidden_units=(64, 64, 64, 64),
):
    """
    Build the final CasADi surrogate u -> z for a given well.

    Pipeline:
        well_name -> load trained PINN + AlgNN -> extract weights/scaling ->
        build CasADi PINN -> build CasADi AlgNN -> compose into u -> z

    Final output z ordering:
        [P_bh_bar, P_tb_b_bar, w_G_inj, w_res, w_L_res, w_G_res, w_w_out, w_o_out]
    """

    # ------------------------------------------------
    # 1. Load well metadata
    # ------------------------------------------------
    wells = get_wells()
    well_list = wells[well_name]

    BSW = well_list["BSW"]
    GOR = well_list["GOR"]

    y_min = well_list["y_min"]
    y_max = well_list["y_max"]
    z_min = well_list["z_min"]
    z_max = well_list["z_max"]

    # ------------------------------------------------
    # 2. Resolve trained model paths
    # ------------------------------------------------
    pinn_path, alg_path = get_model_paths(well_name)

    # ------------------------------------------------
    # 3. Instantiate PyTorch models with correct scaling
    # ------------------------------------------------
    pinn = PINN(
        hidden_units=list(pinn_hidden_units),
        y_min=y_min,
        y_max=y_max,
    )

    algnn = AlgNN(
        hidden_units=list(alg_hidden_units),
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    # ------------------------------------------------
    # 4. Load trained weights
    # ------------------------------------------------
    pinn = load_model_weights(pinn, pinn_path)
    algnn = load_model_weights(algnn, alg_path)

    # ------------------------------------------------
    # 5. Extract raw NN weights
    # ------------------------------------------------
    pinn_w = extract_pinn_standard_weights(pinn)
    alg_w = extract_algnn_standard_weights(algnn)

    # ------------------------------------------------
    # 6. Read scaling constants from trained models
    # ------------------------------------------------
    pinn_y_min = pinn.y_min.cpu().numpy().tolist()
    pinn_y_max = pinn.y_max.cpu().numpy().tolist()
    pinn_u_min = pinn.u_min.cpu().numpy().tolist()
    pinn_u_max = pinn.u_max.cpu().numpy().tolist()

    alg_y_min = algnn.y_min.cpu().numpy().tolist()
    alg_y_max = algnn.y_max.cpu().numpy().tolist()
    alg_u_min = algnn.u_min.cpu().numpy().tolist()
    alg_u_max = algnn.u_max.cpu().numpy().tolist()
    alg_z_min = algnn.z_min.cpu().numpy().tolist()
    alg_z_max = algnn.z_max.cpu().numpy().tolist()

    # ------------------------------------------------
    # 7. Build final CasADi composite surrogate
    # ------------------------------------------------
    F_u2z = build_casadi_surrogate_u2z(
        pinn_weights=pinn_w,
        algnn_weights=alg_w,
        pinn_y_min=pinn_y_min,
        pinn_y_max=pinn_y_max,
        pinn_u_min=pinn_u_min,
        pinn_u_max=pinn_u_max,
        alg_y_min=alg_y_min,
        alg_y_max=alg_y_max,
        alg_u_min=alg_u_min,
        alg_u_max=alg_u_max,
        alg_z_min=alg_z_min,
        alg_z_max=alg_z_max,
        BSW=BSW,
        GOR=GOR,
    )

    return F_u2z


#Garbage dumper

# import casadi as ca
# from application.simulation_engine import make_model
# from utilities.block_builders import *
# from networks.networks import *
#
#
# def optimize_single_well_production_NN(
#         F_u2z,
#         F_pinn,
#         u_guess=(0.5,0.5),
#         P_max_tb_b_bar=120,
#         P_min_bh_bar=90,
#         ):
#     ipopt_opts = {
#         "ipopt.print_level": 0,
#         "print_time": 0,
#         "ipopt.max_iter": 6000,
#         "ipopt.tol": 1e-10,
#         "ipopt.constr_viol_tol": 1e-8,
#         "ipopt.mu_strategy": "adaptive",
#         "ipopt.linear_solver": "mumps",
#     }
#
#     # ---------------------
#     # Decision variable (single well)
#     # ---------------------
#     u = ca.MX.sym("u", 2)  # [u1,u2]
#
#     u1 = u[0]
#     u2 = u[1]
#
#     # Evaluate NN surrogate
#     z = F_u2z(u=u)["z"] if isinstance(F_u2z(u=u), dict) else F_u2z(u)  # robust
#     m_o_out = z[0]
#     P_bh = z[1]
#     P_tb_b = z[2]
#
#     # ---------------------
#     # Objective: maximize oil
#     # ---------------------
#     obj = -m_o_out
#
#     # ---------------------
#     # Constraints
#     #   P_bh >= P_min
#     #   P_tb_b <= P_max
#     # ---------------------
#     b_hat = -0.3268*u1*u1 + 0.5116*u1 + 0.01914
#     g_stab = u2 - b_hat
#
#
#     g = ca.vertcat(P_bh, P_tb_b,g_stab)
#     lbg = ca.DM([float(P_min_bh_bar), -ca.inf,0.0])
#     ubg = ca.DM([ca.inf, float(P_max_tb_b_bar),ca.inf])
#
#     # ---------------------
#     # Bounds / initial guess
#     # ---------------------
#     lbx = ca.DM([0.05, 0.10])
#     ubx = ca.DM([1.0, 1.0])
#     x0 = ca.DM(list(u_guess))
#
#     # ---------------------
#     # Solve NLP
#     # ---------------------
#     nlp = {"x": u, "f": obj, "g": g}
#     solver = ca.nlpsol("single_well_solver", "ipopt", nlp, ipopt_opts)
#
#     sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
#     stats = solver.stats()
#
#     u_star = ca.DM(sol["x"]).full().flatten()
#     y_star = ca.DM(F_pinn(u=u_star)["y"]).full().flatten()
#     z_star = ca.DM(F_u2z(u=u_star)["z"] if isinstance(F_u2z(u=u_star), dict) else F_u2z(u_star)).full().flatten()
#
#     return {
#         "stats": stats,
#         "u_star": u_star,  # [u1*, u2*]
#         "y_star": y_star,
#         "z_star": z_star,  # [m_o_out*, P_bh*, P_tb_b*]
#         "m_o_out": float(z_star[0]),
#         "P_bh": float(z_star[1]),
#         "P_tb_b": float(z_star[2]),
#     }
# #
# #
# from pathlib import Path
# from configuration.wells import get_wells
#
#
#
#
#
#
#
#
# ### Change well name here
# well_name="P2"
# ###########################
# wells=get_wells()
# well_list=wells[well_name]
#
#
# pinn_path, alg_path=get_model_paths(well_name)
#
# BSW=well_list["BSW"]
# GOR=well_list["GOR"]
# PI=well_list["PI"]
# K_gs_sur=well_list["K_gs_sur"]
# K_inj_sur=well_list["K_inj_sur"]
# K_pr_sur=well_list["K_pr_sur"]
# y_guess_sur=well_list["y_guess_sur"]
# y_min=well_list["y_min"]
# y_max=well_list["y_max"]
# z_min=well_list["z_min"]
# z_max=well_list["z_max"]
#
#
# pinn = PINN(hidden_units=[64,64,64],
#             y_min=y_min,
#             y_max=y_max)
#
#
# algnn = AlgNN(hidden_units=[64,64,64,64],
#               y_min=y_min,
#               y_max=y_max,
#               z_min=z_min,
#               z_max=z_max)
#
#
# # 1) Load your torch models
# # from your_code import PINN, AlgNN, load_model_weights
# pinn = load_model_weights(pinn, pinn_path)
# algnn  = load_model_weights(algnn, alg_path)
#
# # 2) Extract weights
# pinn_w = extract_pinn_standard_weights(pinn)
# alg_w  = extract_algnn_standard_weights(algnn)
#
# # 3) IMPORTANT: use the SAME scaling constants as the trained models
# # You can read them from buffers:
# pinn_y_min = pinn.y_min.cpu().numpy().tolist()
# pinn_y_max = pinn.y_max.cpu().numpy().tolist()
# pinn_u_min = pinn.u_min.cpu().numpy().tolist()
# pinn_u_max = pinn.u_max.cpu().numpy().tolist()
#
# alg_y_min = algnn.y_min.cpu().numpy().tolist()
# alg_y_max = algnn.y_max.cpu().numpy().tolist()
# alg_u_min = algnn.u_min.cpu().numpy().tolist()
# alg_u_max = algnn.u_max.cpu().numpy().tolist()
# alg_z_min = algnn.z_min.cpu().numpy().tolist()
# alg_z_max = algnn.z_max.cpu().numpy().tolist()
#
# F_u2z = build_casadi_surrogate_u2z(
#     pinn_weights=pinn_w,
#     algnn_weights=alg_w,
#     pinn_y_min=pinn_y_min, pinn_y_max=pinn_y_max,
#     pinn_u_min=pinn_u_min, pinn_u_max=pinn_u_max,
#     alg_y_min=alg_y_min,   alg_y_max=alg_y_max,
#     alg_u_min=alg_u_min,   alg_u_max=alg_u_max,
#     alg_z_min=alg_z_min,   alg_z_max=alg_z_max,
#     BSW=BSW,
#     GOR=GOR,
# )
#
# # # F_u2z is your chained CasADi surrogate (u -> [m_o_out, P_bh, P_tb_b])
# # res = optimize_single_well_production_NN(
# #     F_u2z=F_u2z,
# #     F_pinn=F_pinn,
# #     u_guess=(0.8, 0.8),
# #     P_min_bh_bar=90.0,
# #     P_max_tb_b_bar=120.0,
# # )
# #
# # print("Solver success:", res["stats"]["success"])
# # print("u* =", res["u_star"])
# # print("y* =", res["y_star"])
# # print("m_o_out* =", res["m_o_out"])
# # print("P_bh* =", res["P_bh"])
# # print("P_tb_b* =", res["P_tb_b"])
# #
# # from solvers.steady_state_solver import solve_equilibrium_ipopt
# # u1=res["u_star"][0]
# # u2=res["u_star"][1]
# #
# # y_guess=res["y_star"]
# #
# # y_guess_rig = [3679.08033973,
# #                289.73390193,
# #                3167.56224658,
# #                1041.96126532,
# #                50.46858403,
# #                759.52720527,
# #                249.84447542]
# #
# # z_guess_rig = [8.75897957e+06,
# # 8.42155186e+06,
# # 2.17230613e+01,
# # 2.17230613e+01]
# #
# # print('\n\n')
# # print(f"Now we take this control and apply to the plant...")
# # model=make_model("rigorous",BSW=0.20,GOR=0.05,PI=3.0e-6)
# # y_star, z_star, dx_star, g_star, out_star, eig, stable, stats= solve_equilibrium_ipopt(
# #     model=model,
# #     u_val=[u1, u2],
# #     y_guess=y_guess_rig,
# #     z_guess=z_guess_rig
# # )
# # print(f"y_star of the model used to train this NN at u*={y_star}")
# # print(f"And the pressure bottomhole is p_bh={out_star[15]} and w_out={out_star[38]}")





















def build_steady_state_model(f_func,
                            state_size,
                            control_size=2,
                            alg_size=None,
                            name="glc_ss",
                            out_name=None,
                            regularize_gz=True,
                            gz_reg=1e-8
                            ):
    """
        Build symbolic CasADi blocks for steady-state models.

        ODE surrogate expected signature:
            dx, out = f_func(y,u)

        DAE rigorous expected signature (alg_size must be provided):
            dx,g,out=f_func(y,z_alg,u)

        Returns a dict. Check model ["is DAE"]

    """
    is_dae=alg_size is not None

    # ---------------------
    # 1) Symbols
    # ---------------------

    y = ca.MX.sym("y", state_size)
    u = ca.MX.sym("u", control_size)

    if is_dae:
        z_alg=ca.MX.sym("z_alg", alg_size)
        z_alg_names=[f"z_alg{i}" for i in range(alg_size)]

        # ---------------------
        # 2) Model evaluation
        # ---------------------

        dx,g,out=f_func(y, z_alg, u)
        # Residual for steady-state solve
        res=ca.vertcat(dx,g)

        # ---------------------
        # 3) Jacobians
        # ---------------------

        yz = ca.vertcat(y, z_alg)
        J_yz = ca.jacobian(res, yz)
        J_u = ca.jacobian(res, u)

        # --- Reduced Jacobian for stability: Ared = fx - fz * (gz \ gx)
        fx = ca.jacobian(dx, y)
        fz = ca.jacobian(dx, z_alg)
        gx = ca.jacobian(g,  y)
        gz = ca.jacobian(g,  z_alg)

        ng = int(g.size1())  # number of algebraic equations
        nz = int(z_alg.size1())

        Ared = ca.MX.nan(fx.size1(), fx.size2())  # default placeholder

        if ng == nz:
            if regularize_gz:
                I = ca.MX.eye(nz)
                X = ca.solve(gz + gz_reg * I, gx)
            else:
                X = ca.solve(gz, gx)
            Ared = fx - fz @ X
        else:
        # Optional: still allow partial "Ared" only if gz is square
            pass


        # ---------------------
        # 4) CasADi Functions
        # ---------------------

        F_all = ca.Function(
            f"{name}_all",
            [y, z_alg, u],
            [dx, g, out],
            ["y", "z_alg", "u"],
            ["dx", "g", "out"],
        )

        F_res = ca.Function(
            f"{name}_res",
            [y, z_alg, u],
            [res],
            ["y", "z_alg", "u"],
            ["res"],
        )

        F_J_yz = ca.Function(
            f"{name}_J_yz",
            [y, z_alg, u],
            [J_yz],
            ["y", "z_alg", "u"],
            ["J_yz"],
        )

        F_J_u = ca.Function(
            f"{name}_J_u",
            [y, z_alg, u],
            [J_u],
            ["y", "z_alg", "u"],
            ["J_u"],
        )

        F_A = ca.Function(
            f"{name}_Ared",
            [y, z_alg, u],
            [Ared],
            ["y", "z_alg", "u"],
            ["A"],
        )



        F_dx_out = ca.Function(
            f"{name}_dx_out",
            [y, z_alg, u],
            [dx, out],
            ["y", "z_alg", "u"],
            ["dx", "out"],
        )

        return {
            # Common
            "is_dae": True,
            "name": name,
            "y": y,
            "u": u,
            "dx": dx,
            "out": out,
            "F_all": F_all,
            "Z_NAMES": out_name,
            "nx": state_size,
            "nu": control_size,
            "F_A":F_A,

            #DAE-specific
            "z_alg": z_alg,
            "g": g,
            "res": res,
            "nz": alg_size,
            "Z_alg_names": z_alg_names,
            "F_res": F_res,
            "F_J_yz": F_J_yz,
            "F_J_u": F_J_u,
            "F_dx_out": F_dx_out,
            "Ared": Ared,
        }
    else:
        # ---------------------
        # 2) Model evaluation
        # ---------------------
        dx,out=f_func(y,u)

        # Stability Jacobian
        A = ca.jacobian(dx, y)

        F_all = ca.Function(
            f"{name}_all",
            [y, u],
            [dx, out],
            ["y", "u"],
            ["dx", "out"])


        F_A = ca.Function(f"{name}_A",
                          [y, u],
                          [A],
                          ["y", "u"],
                          ["A"])

        return {
            # Common
            "is_dae": False,
            "name": name,
            "y": y,
            "u": u,
            "dx": dx,
            "out": out,
            "F_all": F_all,
            "Z_NAMES": out_name,
            "nx": state_size,
            "nu": control_size,
            "F_A": F_A,

            # ODE-specific
            "A": A,

        }

def load_state_dict(model_path: str, device: str = "cpu") -> dict:
    """Loads a PyTorch state_dict from disk (kept simple & robust)."""
    print(f"=== LOADING STATE_DICT FROM {model_path} ===")
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # older torch versions don't support weights_only
        sd = torch.load(model_path, map_location=device)
    if not isinstance(sd, dict):
        raise ValueError("Loaded object is not a state_dict dict.")
    return sd


def infer_pinn_hidden_units_from_state_dict(state_dict: dict) -> list[int]:
    """
    Infers hidden layer sizes for the *standard* (improved_structure=False) PINN.

    Expects keys like:
      layers.0.weight: (h1, n_u)
      layers.1.weight: (h2, h1)
      ...
      out.weight: (n_y, h_last)
    """
    i = 0
    hidden_units = []
    while f"layers.{i}.weight" in state_dict:
        w = state_dict[f"layers.{i}.weight"]
        hidden_units.append(int(w.shape[0]))  # out_features of that layer
        i += 1

    if not hidden_units:
        raise ValueError(
            "Could not infer hidden_units. Expected keys like 'layers.0.weight' in state_dict."
        )
    return hidden_units


def rebuild_pinn_from_weights(model_path: str,
                              y_min: list,
                              y_max: list,
                              device: str = "cpu") -> PINN:
    """
    Rebuilds your PINN architecture from the saved weights, then loads the weights.
    Assumes improved_structure=False.
    """
    sd = load_state_dict(model_path, device=device)

    hidden_units = infer_pinn_hidden_units_from_state_dict(sd)

    # Infer input/output sizes from weight shapes
    n_u = int(sd["layers.0.weight"].shape[1])
    n_y = int(sd["out.weight"].shape[0])

    # Build model with inferred architecture.
    # Scaling buffers (u_min/u_max/y_min/y_max) will be overwritten by load_state_dict.
    model = PINN(hidden_units=hidden_units,
                 y_min=y_min,
                 y_max=y_max,
                 n_u=n_u,
                 n_y=n_y,
                 improved_structure=False).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Rebuilt PINN: n_u={n_u}, n_y={n_y}, hidden_units={hidden_units}")
    return model

# ---------------------------
# AlgNN rebuild (standard)
# ---------------------------
def infer_algnn_hidden_units_from_state_dict(state_dict: dict) -> list[int]:
    """
    Infers AlgNN hidden sizes from:
      main_layers.0.weight: (h1, n_in)
      main_layers.1.weight: (h2, h1)
      ...
      output_layer.weight: (n_out, h_last)
    """
    i = 0
    hidden_units = []
    while f"main_layers.{i}.weight" in state_dict:
        w = state_dict[f"main_layers.{i}.weight"]
        hidden_units.append(int(w.shape[0]))
        i += 1
    if not hidden_units:
        raise ValueError("Could not infer AlgNN hidden_units (expected keys like 'main_layers.0.weight').")
    return hidden_units


def rebuild_algnn_from_weights(model_path: str,
                               y_min: list,
                               y_max: list,
                               z_min: list,
                               z_max: list,
                               device: str = "cpu") -> AlgNN:

    sd = load_state_dict(model_path, device=device)

    hidden_units = infer_algnn_hidden_units_from_state_dict(sd)

    # Build model. Buffers (y_min/y_max/u_min/u_max/z_min/z_max) are overwritten by load_state_dict anyway.
    model = AlgNN(hidden_units=hidden_units,
                  y_min=y_min,
                  y_max=y_max,
                  z_min=z_min,
                  z_max=z_max).to(device)

    model.load_state_dict(sd)
    model.eval()

    n_out = int(sd["output_layer.weight"].shape[0])
    print(f"Rebuilt AlgNN: n_out={n_out}, hidden_units={hidden_units}")
    return model
