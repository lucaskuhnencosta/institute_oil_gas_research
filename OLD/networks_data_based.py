import torch
import torch.nn as nn

class SteadyMLP(nn.Module):
    def __init__(self, hidden_units=(64, 64, 64), n_in=2, n_out=3, activation=nn.Tanh):
        super().__init__()
        sizes = [n_in] + list(hidden_units)
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(hidden_units))])
        self.out = nn.Linear(sizes[-1], n_out)
        self.act = activation()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        a = u
        for layer in self.layers:
            a = self.act(layer(a))
        return self.out(a)