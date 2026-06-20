import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, batch_size, deter_dim, discrete_dim):
        super().__init__()
        self.batch_size = batch_size
        self.total_dim = deter_dim + discrete_dim
        self.stride = 2

        self.convs == nn.Sequential(
            nn.Linear(in_features = self.embed_dim, out_features = 4096),
            nn.Unflatten(-1, (256,4,4)),
            nn.SiLU(),
            nn.Conv2d(256, 128, 3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, stride = self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(64,32,3, stride=self.stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, stride = self.stride, padding=1)
        )


    def forward(self, h, z):   
        input = torch.cat((h,z), dim=-1)
        x = self.convs(input)
