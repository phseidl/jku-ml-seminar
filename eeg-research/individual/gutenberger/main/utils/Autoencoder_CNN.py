import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=18):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: 3 1D convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),  # [batch_size, 64, 256]
            #nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),             # [batch_size, 128, 128]
            #nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),            # [batch_size, 256, 64]
            #nn.ReLU()
        )
        
        # Decoder: 3 1D transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # [batch_size, 128, 128]
            #nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),   # [batch_size, 64, 256]
            #nn.ReLU(),
            nn.ConvTranspose1d(64, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # [batch_size, input_channels, 512]
            #nn.Sigmoid()  # Sigmoid for output normalization between 0 and 1
        )

    def forward(self, x):
        x = x.squeeze(1)
        # Forward pass through encoder
        x = self.encoder(x)
        
        # Forward pass through decoder
        x = self.decoder(x)
        
        return x.unsqueeze(1)
    

class ConvAutoencoder_Compress(nn.Module):
    def __init__(self, input_channels=18):
        super(ConvAutoencoder_Compress, self).__init__()
        
        # Encoder: 3 1D convolutional layers
        self.c1 = nn.Conv2d(1, 18, kernel_size=3, stride=2, padding=0, dilation=1)
        self.c2 = nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, dilation=1)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 18, kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=0, dilation=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        )

        
        # Decoder: 3 1D transposed convolutional layers
        self.decoder = nn.Sequential(
            # Transpose Conv 1: Upsample from (2, 512) to (4, 512)
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0, output_padding=0),
            # Transpose Conv 2: Upsample from (4, 512) to (9, 512)
            nn.ConvTranspose2d(32, 18, kernel_size=3, stride=2, padding=0, output_padding=(1,0)),
            # Transpose Conv 3: Upsample from (9, 512) to (18, 512)
            nn.ConvTranspose2d(18, 1, kernel_size=3, stride=2, padding=0, output_padding=1)  # Adjust output_padding to match
        )

    def forward(self, x):
        # Forward pass through encoder
        x = self.encoder(x)
        
        # Forward pass through decoder
        x = self.decoder(x)
        
        return x