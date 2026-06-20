import torch.nn as nn
import torch

from encoder import Encoder


# Legend: 
#   e = encoded state of the environment
#   h = recurrent state 
#   z = discrete representation of current state
#   a = action vector

class Posterior(nn.Module):
    def __init__(self, embed_dim, deter_dim):
        super().__init__()
        self.embed_dim = embed_dim  
        self.deter_dim = deter_dim
        self.total_dim = embed_dim + deter_dim
        self.hidden_dim = 512               #TODO: un-hardcode
        self.stoch_dim = 512                #TODO: un-hardcode

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim)
        )
        

    def forward(self, e, h):
        z = torch.cat((e, h), dim=-1)
        z = self.mlp(z)
        return z
    


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
        z = self.mlp(h)
        return z
        
class SequenceModel(nn.Module):     #Not true to paper, this uses a regular GRUcell, not the block diagonal version
    def __init__(self, deter_dim, discrete_dim, action_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim = action_dim
        self.hidden_dim = 512

        self.linear = nn.Linear(self.discrete_dim + self.action_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)

    def forward(self, h, z, a):
        input = torch.cat((z,a), dim=-1)
        x = self.linear(input)
        h_new = self.gru(x, h)
        return h_new

class RSSM(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = SequenceModel()
        self.prior = Prior()
        self.posterior = Posterior()
        self.encoder = Encoder()
    
    def initial(self, batch_size):
        return

    def observation_step(self, h, z, a, e):
        return

    def imagination_step(self, h, z, a):
        return z