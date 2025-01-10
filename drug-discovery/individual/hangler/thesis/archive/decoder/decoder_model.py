import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Dict
import logging as logger

class LSTMDecoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        decoder_layers: int = 1,
        lr: float = 1e-3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.decoder_layers = decoder_layers
        self.lr = lr

        # Embeddings for tokens in the output space
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=decoder_layers,
            batch_first=True
        )

        # Projection to vocab
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # A linear projection from embedding vector to initial hidden state
        self.init_hidden_proj = nn.Linear(embedding_dim, hidden_dim * decoder_layers)
        self.init_cell_proj = nn.Linear(embedding_dim, hidden_dim * decoder_layers)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, context_embeddings: torch.Tensor, input_ids: torch.Tensor):
        """
        Forward pass of the decoder.
        Args:
            context_embeddings: (batch, embedding_dim)
            input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.size()

        # Initialize hidden, cell states from context
        h_0 = self.init_hidden_proj(context_embeddings)  # (batch, hidden_dim * layers)
        c_0 = self.init_cell_proj(context_embeddings)

        # Reshape for (layers, batch, hidden_dim)
        h_0 = h_0.view(self.decoder_layers, batch_size, self.hidden_dim).contiguous()
        c_0 = c_0.view(self.decoder_layers, batch_size, self.hidden_dim).contiguous()

        # Embed tokens
        token_emb = self.token_embeddings(input_ids)  # (batch, seq_len, hidden_dim)

        # Pass through LSTM
        outputs, _ = self.lstm(token_emb, (h_0, c_0))  # (batch, seq_len, hidden_dim)

        # Project to vocab logits
        logits = self.output_proj(outputs)  # (batch, seq_len, vocab_size)
        return logits

    def training_step(self, batch: Any, batch_idx: int):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        # Accuracy calculation
        preds = torch.argmax(logits, dim=-1)
        mask = target_ids != 0
        correct = (preds == target_ids) & mask
        acc = correct.sum() / mask.sum()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        preds = torch.argmax(logits, dim=-1)
        mask = target_ids != 0
        correct = (preds == target_ids) & mask
        acc = correct.sum() / mask.sum()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)

        preds = torch.argmax(logits, dim=-1)
        mask = target_ids != 0
        correct = (preds == target_ids) & mask
        acc = correct.sum() / mask.sum()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
