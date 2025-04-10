import torch
import torch.nn as nn
import torch.nn.functional as F

# Permute2d Layer 
class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)

# CLEEGN Autoencoder with Latent Extraction
class CLEEGN2Vec(nn.Module):
    def __init__(self, n_chan, fs, N_F=20, tem_kernelLen=0.1):
        super(CLEEGN2Vec, self).__init__()
        self.n_chan = n_chan
        self.N_F = N_F
        self.fs = fs

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, n_chan, (n_chan, 1), padding="valid", bias=True),
            Permute2d((0, 2, 1, 3)),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.99)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(1, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(N_F, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(N_F, n_chan, (n_chan, 1), padding="same", bias=True),
            nn.BatchNorm2d(n_chan, eps=1e-3, momentum=0.99)
        )
        self.decoder_conv3 = nn.Conv2d(n_chan, 1, (n_chan, 1), padding="same", bias=True)

    def encode(self, x):
        # Encoder Path
        x = self.encoder_conv1(x)
        latent = self.encoder_conv2(x)
        return latent  # Latent representation

    def decode(self, latent):
        # Decoder Path
        x = self.decoder_conv1(latent)
        x = self.decoder_conv2(x)
        reconstruction = self.decoder_conv3(x)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent