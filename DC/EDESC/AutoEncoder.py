import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear

class AE(nn.Module):

    def __init__(self, n_enc_1, n_dec_1, n_input, n_z):
        super(AE, self).__init__()

        # Encoder
        self.enc_1 = Linear(n_input, n_enc_1)

        self.z_layer = Linear(n_enc_1, n_z)

        # Decoder
        self.dec_1 = Linear(n_z, n_dec_1)

        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):

        # Encoder
        enc_h1 = F.relu(self.enc_1(x))

        z = self.z_layer(enc_h1)

        # Decoder
        dec_h1 = F.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z
