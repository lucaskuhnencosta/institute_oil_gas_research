import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Vanilla steady-state PINN network:
    maps controls u -> states y_hat.

    u: (..., n_u)
    returns y_hat: (..., n_y)
    """

    def __init__(
        self,
        hidden_units: list[int],
        n_u: int = 2,
        n_y: int = 3,
        y_min: list[float] = [3032.55, 220.05, 6341.30],
        y_max: list[float] = [4796.20, 1094.60, 11990.90],
        u_min: list[float] = [0.05, 0.10],
        u_max: list[float] = [1.0, 1.0],
        improved_structure: bool = False,
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()

        if not hidden_units:
            raise ValueError("hidden_units must be a non-empty list, e.g., [64,64,64].")

        self.n_u = n_u

        self.improved_structure = improved_structure

        self.act = activation()

        # store u scaling
        self.register_buffer("u_min", torch.tensor(u_min, dtype=torch.float32))
        self.register_buffer("u_max", torch.tensor(u_max, dtype=torch.float32))
        self.register_buffer("y_min", torch.tensor(y_min, dtype=torch.float32))
        self.register_buffer("y_max", torch.tensor(y_max, dtype=torch.float32))
        self.n_y = len(y_min)

        if improved_structure:
            self.encoder_1 = nn.Linear(self.n_u, hidden_units[0])
            self.encoder_2 = nn.Linear(self.n_u, hidden_units[0])

            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(self.n_u, hidden_units[0]))
            for i in range(1, len(hidden_units)):
                self.layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))

            self.out = nn.Linear(hidden_units[-1], self.n_y)
        else:
            sizes = [n_u] + hidden_units
            self.layers = nn.ModuleList(
                [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_units))]
            )
            self.out = nn.Linear(hidden_units[-1], self.n_y)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _scale_u(self, u: torch.Tensor) -> torch.Tensor:
        # u_scaled in [-1, 1]
        return 2.0 * (u - self.u_min) / (self.u_max - self.u_min + 1e-12) - 1.0

    def _descale_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_squash = torch.tanh(y_norm)  # in [-1,1]
        return 0.5 * (y_squash + 1.0) * (self.y_max - self.y_min) + self.y_min

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (..., n_u)
        returns y_hat: (..., n_y)
        """
        x = self._scale_u(u)

        if self.improved_structure:
            e1 = self.act(self.encoder_1(x))
            e2 = self.act(self.encoder_2(x))
            a = x
            for layer in self.layers:
                z = self.act(layer(a))
                a = z * e1 + (1.0 - z) * e2
            y_norm = self.out(a)
        else:
            a = x
            for layer in self.layers:
                a = self.act(layer(a))
            y_norm = self.out(a)

        y_hat = self._descale_y(y_norm)
        return y_hat


class AlgNN(nn.Module):
    def __init__(self,
                 hidden_units: list,
                 y_min: list[float] = [3032.55, 220.05, 6341.30],
                 y_max: list[float] = [4796.20, 1094.60, 11990.90],
                 u_min: list[float] = [0.05, 0.10],
                 u_max: list[float] = [1.0, 1.0],
                 z_min: list[float] = [3.50,84.95,79.0],  # Output de-normalization
                 z_max: list[float] =[17.15,144.75,138.77],
                 Nonlin=nn.Tanh) -> None:
        super().__init__()

        # --- Register all normalization constants as buffers ---

        # Input scaling
        self.register_buffer('y_min', torch.tensor(y_min, dtype=torch.float32))
        self.register_buffer('y_max', torch.tensor(y_max, dtype=torch.float32))
        self.register_buffer('u_min', torch.tensor(u_min, dtype=torch.float32))
        self.register_buffer('u_max', torch.tensor(u_max, dtype=torch.float32))

        # Output scaling
        self.register_buffer('z_min', torch.tensor(z_min, dtype=torch.float32))
        self.register_buffer('z_max', torch.tensor(z_max, dtype=torch.float32))

        # Pre-calculate ranges for efficiency
        self.register_buffer('y_range', self.y_max - self.y_min)
        self.register_buffer('u_range', self.u_max - self.u_min)
        self.register_buffer('z_range', self.z_max - self.z_min)

        # --- Define network dimensions ---
        n_in = len(y_min) + len(u_min)
        n_out = len(z_min)

        # --- Build the network layers (standard FNN) ---
        self.main_layers = nn.ModuleList()
        layer_sizes = [n_in] + hidden_units

        for i in range(len(hidden_units)):
            self.main_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_units[-1], n_out)
        self.activation = Nonlin()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use xavier_normal_ (Glorot Normal) for weights
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # Use xavier_normal_ for biases as well
                    nn.init.zeros_(m.bias)

    def forward(self, y, u):
        y_scaled = 2 * (y - self.y_min) / self.y_range - 1
        u_scaled = 2 * (u - self.u_min) / self.u_range - 1
        input_tensor = torch.cat([y_scaled, u_scaled], dim=-1)
        a = input_tensor
        for layer in self.main_layers:
            a = self.activation(layer(a))
        output_normalized = self.output_layer(a)
        output_physical = ((output_normalized + 1) / 2) * self.z_range + self.z_min

        return output_physical