import torch.nn as nn
import torch

class Prior(nn.Module):
    def __init__(self, deter_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.hidden_dim = 512          #TODO:un-hardcode
        self.stoch_dim = 512            #TODO:un-hardcode

        self.mlp = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim)
        )

    def forward(self, h):
        z = self.mlp(e)
        return z
        