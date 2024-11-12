import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, target=None):
        x = x.squeeze(1).permute(0, 2, 1)
        if target is not None:
            target = target.squeeze(1).permute(0, 2, 1)
        batch_size, seq_len, _ = x.size()
        
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, seq_len, x.size(2)).to(x.device)
        
        # Decoder initial input (first step of the sequence)
        decoder_input = x[:, 0, :].unsqueeze(1)  # shape: [batch_size, 1, input_dim]
        
        for t in range(seq_len):
            # Decode step-by-step
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Apply the final dense layer to map to the output dimension
            out = self.fc(out)
            
            # Save the output for this time step
            outputs[:, t, :] = out.squeeze(1)
            
            # Decide if we use teacher forcing
            if target is not None and random.random() < self.teacher_forcing_ratio:
                # Use the actual target (clean data) as the next input
                decoder_input = target[:, t, :].unsqueeze(1)
            else:
                # Use the model's own prediction as the next input
                decoder_input = out
        
        return outputs.permute(0, 2, 1).unsqueeze(1)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Encoding the noisy EEG sequence
        x = x.squeeze(1).permute(0, 2, 1)
        _, (hidden, cell) = self.encoder(x)

        #decoder_input = torch.zeros(x.size(0), x.size(1), hidden.size(2))
        decoder_input = x
        # Decoding (using the same sequence length for input and output)
        out, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Final dense layer to get the output shape [batch, 512, 18]
        out = self.fc(out)

        out = out.permute(0, 2, 1).unsqueeze(1)
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # Ensure shape [batch_size, sequence_length, input_dim]
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = out.permute(0, 2, 1).unsqueeze(1)
        return out
