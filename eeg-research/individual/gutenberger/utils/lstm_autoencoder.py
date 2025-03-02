import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)  # Compress to latent_dim

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Only keep the last hidden state
        latent = self.fc(hidden[-1])   # Map hidden state to latent space
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)  # Map latent space back to hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, seq_len):
        # Expand latent vector to sequence for LSTM decoding
        x = self.fc(x).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Decode sequence
        output, _ = self.lstm(x)
        output = self.output_layer(output)  # Map to output_dim
        return output

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        seq_len = x.size(1)  # Length of the sequence
        latent = self.encoder(x)
        output = self.decoder(latent, seq_len)
        return output.permute(0, 2, 1).unsqueeze(1)



