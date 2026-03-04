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
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()

        if not hidden_units:
            raise ValueError("hidden_units must be a non-empty list, e.g., [64,64,64].")

        self.n_u = n_u
        self.n_y = n_y
        self.act = activation()

        sizes = [n_u] + hidden_units
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_units))]
        )
        self.out = nn.Linear(hidden_units[-1], n_y)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (..., n_u)
        returns y_hat: (..., n_y)
        """
        a = u
        for layer in self.layers:
            a = self.act(layer(a))
        y_hat = self.out(a)
        return y_hat