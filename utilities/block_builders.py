import casadi as ca
from networks.networks import PINN
from networks.networks import AlgNN
import torch

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
    y_min_ca = ca.DM(y_min); y_max_ca = ca.DM(y_max)
    u_min_ca = ca.DM(u_min); u_max_ca = ca.DM(u_max)

    y_range = y_max_ca - y_min_ca
    u_range = u_max_ca - u_min_ca

    n_u = len(u_min)
    n_y = len(y_min)

    u_sym = ca.MX.sym("u", n_u)

    # u scaling (same as PINN._scale_u)
    u_scaled = 2 * (u_sym - u_min_ca) / (u_range + 1e-12) - 1
    a = u_scaled

    # hidden layers
    for W_np, b_np in zip(weights_dict["hidden_layers_w"], weights_dict["hidden_layers_b"]):
        W = ca.DM(W_np)
        b = ca.DM(b_np).reshape((-1, 1))
        a = activation(W @ a + b)

    # output
    out_w = ca.DM(weights_dict["output_w"])
    out_b = ca.DM(weights_dict["output_b"]).reshape((-1, 1))
    y_norm = out_w @ a + out_b

    # y de-scale (same as PINN._descale_y)
    y_squash = ca.tanh(y_norm)
    y_hat = 0.5 * (y_squash + 1.0) * y_range + y_min_ca

    return ca.Function("F_pinn_u2y", [u_sym], [y_hat], ["u"], ["y"])


def build_casadi_algnn_function_standard(
    weights_dict: dict,
    y_min: list, y_max: list,
    u_min: list, u_max: list,
    z_min: list, z_max: list,
    activation=ca.tanh
):
    y_min_ca = ca.DM(y_min); y_max_ca = ca.DM(y_max)
    u_min_ca = ca.DM(u_min); u_max_ca = ca.DM(u_max)
    z_min_ca = ca.DM(z_min); z_max_ca = ca.DM(z_max)

    y_range = y_max_ca - y_min_ca
    u_range = u_max_ca - u_min_ca
    z_range = z_max_ca - z_min_ca

    n_y = len(y_min)
    n_u = len(u_min)
    n_z = len(z_min)

    y_sym = ca.MX.sym("y", n_y)
    u_sym = ca.MX.sym("u", n_u)

    y_scaled = 2 * (y_sym - y_min_ca) / (y_range + 1e-12) - 1
    u_scaled = 2 * (u_sym - u_min_ca) / (u_range + 1e-12) - 1
    a = ca.vertcat(y_scaled, u_scaled)

    for W_np, b_np in zip(weights_dict["hidden_layers_w"], weights_dict["hidden_layers_b"]):
        W = ca.DM(W_np)
        b = ca.DM(b_np).reshape((-1, 1))
        a = activation(W @ a + b)

    out_w = ca.DM(weights_dict["output_w"])
    out_b = ca.DM(weights_dict["output_b"]).reshape((-1, 1))
    z_norm = out_w @ a + out_b

    z_hat = 0.5 * (z_norm + 1.0) * z_range + z_min_ca

    return ca.Function("F_algnn_yu2z", [y_sym, u_sym], [z_hat], ["y", "u"], ["z"])

def build_casadi_surrogate_u2z(
    pinn_weights: dict,
    algnn_weights: dict,
    pinn_y_min: list, pinn_y_max: list,
    pinn_u_min: list, pinn_u_max: list,
    alg_y_min: list, alg_y_max: list,
    alg_u_min: list, alg_u_max: list,
    alg_z_min: list, alg_z_max: list,
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
        z_min=alg_z_min, z_max=alg_z_max
    )

    u = ca.MX.sym("u", len(pinn_u_min))
    y = F_pinn(u=u)["y"]              # (3x1)
    z = F_alg(y=y, u=u)["z"]          # (3x1): [m_o_out, p_bh, p_tb_b] depending on your z_min/z_max order

    return ca.Function("F_u2z", [u], [z], ["u"], ["z"])



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


def rebuild_pinn_from_weights(model_path: str, device: str = "cpu") -> PINN:
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
    model = PINN(hidden_units=hidden_units, n_u=n_u, n_y=n_y, improved_structure=False).to(device)
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


def rebuild_algnn_from_weights(model_path: str, device: str = "cpu") -> AlgNN:
    sd = load_state_dict(model_path, device=device)

    hidden_units = infer_algnn_hidden_units_from_state_dict(sd)

    # Build model. Buffers (y_min/y_max/u_min/u_max/z_min/z_max) are overwritten by load_state_dict anyway.
    model = AlgNN(hidden_units=hidden_units).to(device)
    model.load_state_dict(sd)
    model.eval()

    n_out = int(sd["output_layer.weight"].shape[0])
    print(f"Rebuilt AlgNN: n_out={n_out}, hidden_units={hidden_units}")
    return model
