import torch
import torch.nn as nn

class PINN(nn.Module):

    def __init__(self,
                 hidden_units: list[int],
                 n_u: int = 2,
                 n_y: int = 3,
                 activation: type[nn.Module] = nn.Tanh,
                 improved_structure: bool = False):
        super().__init__()
        self.improved_structure = improved_structure
        act = activation


        n_in = n_y + n_u
        n_out = n_y

        if not hidden_units:
            raise ValueError("hidden_units must be a non-empty list, e.g., [64,64,64].")

        if improved_structure:
            # simple gated/skip style similar to what you had (but for static u->y)
            self.encoder_1 = nn.Linear(n_in, hidden_units[0])
            self.encoder_2 = nn.Linear(n_in, hidden_units[0])

            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(n_in, hidden_units[0]))
            for i in range(1, len(hidden_units)):
                self.layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))

            self.out = nn.Linear(hidden_units[-1], n_out)
            self.act = act()
        else:
            sizes = [n_in] + hidden_units
            self.layers = nn.ModuleList(
                [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden_units))]
            )
            self.out = nn.Linear(hidden_units[-1], n_out)
            self.act = act()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                y: torch.Tensor,
                u: torch.Tensor) -> torch.Tensor:
        """
        u: (..., 2)
        returns y_hat: (..., 3)
        """

        x = torch.cat([y, u], dim=-1)  # (..., 5)
        a = x
        if self.improved_structure:
            e1 = self.act(self.encoder_1(a))
            e2 = self.act(self.encoder_2(a))
            for layer in self.layers:
                z = self.act(layer(a))
                a = z * e1 + (1.0 - z) * e2
            y_hat = self.out(a)
        else:
            for layer in self.layers:
                a = self.act(layer(a))
            y_hat = self.out(a)
        return y_hat