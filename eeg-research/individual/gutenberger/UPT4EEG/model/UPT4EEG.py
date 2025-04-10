""" from https://github.com/BenediktAlkin/upt-tutorial/blob/main/upt/models/upt_sparseimage_autoencoder.py """

from torch import nn

class UPT4EEG(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_feat, input_pos, batch_idx, output_pos):
        # encode data
        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            batch_idx=batch_idx,
        )

        # decode
        pred = self.decoder(latent, output_pos=output_pos)

        return pred