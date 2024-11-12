import torch
import torch.nn as nn

class LSTMConvAutoencoder(nn.Module):
    def __init__(self, input_channels=18, sequence_length=512, hidden_dim=64, latent_dim=128):
        super(LSTMConvAutoencoder, self).__init__()
        
        # LSTM Encoder for Temporal Information
        self.lstm_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # 2D Convolutional Encoder for Spatial Information
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # [batch_size, 64, sequence_length/2, input_channels/2]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),             # [batch_size, 128, sequence_length/4, input_channels/4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),            # [batch_size, 256, sequence_length/8, input_channels/8]
            nn.ReLU()
        )
        
        # Fully Connected Layers for Latent Space
        conv_out_dim = 49152 #256 * (sequence_length // 8) * (input_channels // 8)
        self.fc1 = nn.Linear(conv_out_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, conv_out_dim)
        
        # 2D Convolutional Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2 * hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU()
        )
        
        # LSTM Decoder for Temporal Reconstruction
        self.lstm_decoder = nn.LSTM(input_size=2 * hidden_dim, hidden_size=1, num_layers=2, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        x = x.squeeze(1)
        # Initial input shape: [batch_size, input_channels, sequence_length]
        
        # 1. Reshape to apply LSTM per channel: [batch_size * input_channels, sequence_length, 1]
        batch_size, input_channels, sequence_length = x.size()
        x = x.view(batch_size * input_channels, sequence_length, 1)
        
        # 2. LSTM Encoder: Capturing temporal information
        x, _ = self.lstm_encoder(x)
        x = x[:, :, :self.lstm_encoder.hidden_size] + x[:, :, self.lstm_encoder.hidden_size:]  # Sum bidirectional outputs
        x = x.view(batch_size, input_channels, sequence_length, -1).permute(0, 3, 2, 1)  # [batch_size, 2*hidden_dim, sequence_length, input_channels]
        
        # 3. 2D Convolutional Encoder for Spatial Features
        x = self.encoder_conv(x)
        
        # 4. Flatten and pass through fully connected layer for latent space compression
        x = x.view(x.size(0), -1)
        x = self.fc1(x)            # Compress to latent space
        x = self.fc2(x)            # Expand back
        
        # 5. Reshape and pass through 2D Convolutional Decoder
        x = x.view(batch_size, 256, sequence_length // 8, input_channels // 8)
        x = self.decoder_conv(x)
        
        # 6. Reshape to [batch_size * input_channels, sequence_length, hidden_dim] for LSTM Decoder
        x = x.permute(0, 3, 2, 1).contiguous()  # [batch_size, input_channels, sequence_length, 2*hidden_dim]
        x = x.view(batch_size * input_channels, sequence_length, -1)
        
        # 7. LSTM Decoder: Reconstructing temporal sequence for each channel
        x, _ = self.lstm_decoder(x)
        x = x[:, :, :self.lstm_decoder.hidden_size] + x[:, :, self.lstm_decoder.hidden_size:]  # Sum bidirectional outputs
        x = x.view(batch_size, input_channels, sequence_length)  # Reshape to original shape
        
        return x
    



class LSTMConvAutoencoder2(nn.Module):
    def __init__(self, input_dim=18, hidden_dim = 18, num_layers = 1):
        super(LSTMConvAutoencoder2, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Encoder: 3 1D convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),  # [batch_size, 64, 256]
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
            nn.ConvTranspose1d(64, input_dim, kernel_size=5, stride=2, padding=2, output_padding=1),  # [batch_size, input_channels, 512]
            #nn.Sigmoid()  # Sigmoid for output normalization between 0 and 1
        )

    def forward(self, x):
        x = x.squeeze(1).permute(0,2,1)

        x, _ = self.lstm(x)  #lstm takes [batch_size, sequence_length, input_dim]
        x = x.permute(0,2,1)
        # Forward pass through encoder
        x = self.encoder(x)                #encoder takes [batch_size, input_dim, sequence_length]
        
        # Forward pass through decoder
        x = self.decoder(x)
        
        return x.unsqueeze(1)
    

class LSTMConvAutoencoder3(nn.Module):
    def __init__(self, input_dim=18, num_layers = 1):
        super(LSTMConvAutoencoder3, self).__init__()
        
        self.lstm = nn.LSTM(256, 256, num_layers, batch_first=True)

        # Encoder: 3 1D convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),  # [batch_size, 64, 256]
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
            nn.ConvTranspose1d(64, input_dim, kernel_size=5, stride=2, padding=2, output_padding=1),  # [batch_size, input_channels, 512]
            #nn.Sigmoid()  # Sigmoid for output normalization between 0 and 1
        )

    def forward(self, x):
        x = x.squeeze(1)

        # Forward pass through encoder
        x = self.encoder(x)                #encoder takes [batch_size, input_dim, sequence_length]
        x, _ = self.lstm(x.permute(0,2,1))  
        # Forward pass through decoder
        x = self.decoder(x.permute(0,2,1))
        
        return x.unsqueeze(1)
    


class LSTMConvAutoencoder4(nn.Module):
    def __init__(self, input_dim=18, num_layers = 1):
        super(LSTMConvAutoencoder4, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)

        # Encoder: 3 1D convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),  # [batch_size, 64, 256]
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
            nn.ConvTranspose1d(64, input_dim, kernel_size=5, stride=2, padding=2, output_padding=1),  # [batch_size, input_channels, 512]
            #nn.Sigmoid()  # Sigmoid for output normalization between 0 and 1
        )

    def forward(self, x):
        x = x.squeeze(1)

        # Forward pass through encoder
        x = self.encoder(x)                #encoder takes [batch_size, input_dim, sequence_length]

        # Forward pass through decoder
        x = self.decoder(x)

        x, _ = self.lstm(x.permute(0,2,1))      #lstm takes [batch_size, sequence_length, input_dim]
        return x.permute(0,2,1).unsqueeze(1)