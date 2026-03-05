import casadi as ca
from Networks.networks import PINN
from Networks.networks import AlgNN
import torch

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
