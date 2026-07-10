import torch.nn as nn
import torch.nn.functional as F
import torch

from encoder import Encoder
from decoder import Decoder

# Legend:
#   e = encoded observation             (output of Encoder)
#   h = deterministic recurrent state   (output of GRU)
#   z = stochastic discrete latent state    (output of Dynamics/Representation)
#   a = action vector

# Representation model
class Posterior(nn.Module):
    def __init__(self, embed_dim, deter_dim, discrete_dim, hidden_dim=512):
        super().__init__()
        self.total_dim = embed_dim + deter_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, discrete_dim)  # outputs flat logits (B, num_cats*cat_size)
        )

    def forward(self, e, h):
        return self.mlp(torch.cat((e, h), dim=-1))


# Dynamics predictor
class Prior(nn.Module):
    def __init__(self, deter_dim, discrete_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, discrete_dim)  # outputs flat logits (B, num_cats*cat_size)
        )

    def forward(self, h):
        return self.mlp(h)


class SequenceModel(nn.Module):     # uses a regular GRUCell, not the block-diagonal variant from the paper
    def __init__(self, action_dim, deter_dim, discrete_dim, hidden_dim=512):
        super().__init__()
        self.linear = nn.Linear(discrete_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

    def forward(self, h, z, a):
        x = self.linear(torch.cat((z, a), dim=-1))
        return self.gru(x, h)


class RSSM(nn.Module):

    def __init__(self, embed_dim, hidden_dim, deter_dim, discrete_dim, action_dim,
                 num_cats=32, in_channels=3, obs_size=64):
        super().__init__()
        # discrete_dim is the total flat z size (num_cats * cat_size)
        # e.g. discrete_dim=1024, num_cats=32 → cat_size=32
        assert discrete_dim % num_cats == 0, "discrete_dim must be divisible by num_cats"
        self.num_cats    = num_cats
        self.cat_size    = discrete_dim // num_cats
        self.deter_dim   = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim  = action_dim

        self.sequence       = SequenceModel(action_dim, deter_dim, discrete_dim, hidden_dim)
        self.dynamics       = Prior(deter_dim, discrete_dim, hidden_dim)
        self.representation = Posterior(embed_dim, deter_dim, discrete_dim, hidden_dim)
        self.encoder        = Encoder(embed_dim, in_channels=in_channels, obs_size=obs_size)
        self.decoder        = Decoder(deter_dim, discrete_dim, out_channels=in_channels)

    def _sample_z(self, logits):
        """Straight-through one-hot from flat logits (B, num_cats*cat_size)."""
        B = logits.shape[0]
        logits_2d = logits.reshape(B, self.num_cats, self.cat_size)
        soft = F.softmax(logits_2d, dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=self.cat_size).float()
        # straight-through: hard forward, soft gradients backward
        z = (hard - soft.detach() + soft)
        return z.reshape(B, self.discrete_dim)

    def initial(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.discrete_dim, device=device)
        return h, z

    def observation_step(self, h, z, a, e):
        h_new        = self.sequence(h, z, a)
        post_logits  = self.representation(e, h_new)   # posterior: uses real obs
        prior_logits = self.dynamics(h_new)             # prior: no obs
        z_new        = self._sample_z(post_logits)
        return h_new, z_new, prior_logits, post_logits

    def imagination_step(self, h, z, a):
        h_new        = self.sequence(h, z, a)
        prior_logits = self.dynamics(h_new)
        z_new        = self._sample_z(prior_logits)
        return h_new, z_new

    def kl_loss(self, prior_logits, post_logits, free_bits=1.0):
        B = post_logits.shape[0]
        post  = F.softmax(post_logits.reshape(B, self.num_cats, self.cat_size), dim=-1)
        prior = F.softmax(prior_logits.reshape(B, self.num_cats, self.cat_size), dim=-1)
        # KL per category (B, num_cats), sum over cats, mean over batch
        kl = (post * (post.log() - prior.log())).sum(-1).sum(-1).mean()
        return torch.clamp(kl, min=free_bits)

    def decode(self, h, z):
        return self.decoder(h, z)
