import torch
import pytorch_lightning as pl
from torch import nn
import logging as logger

class TransformerDecoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        decoder_layers: int = 2,
        lr: float = 2e-3,
        pad_idx: int = 0,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = decoder_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_idx = pad_idx

        # Token embedding
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)
        # Positional encoding (simple trainable version for demonstration)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1024, self.hidden_dim))

        # Linear projections for context
        self.context_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, context_embeddings: torch.Tensor, input_ids: torch.Tensor):
        """
        context_embeddings: (batch, embedding_dim)
        input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.size()

        # (batch, seq_len, hidden_dim)
        token_emb = self.token_embeddings(input_ids)
        # Add positional encodings (simple slice)
        token_emb = token_emb + self.pos_encoding[:, :seq_len, :]

        # Project context to memory shape: (batch, 1, hidden_dim)
        memory = self.context_proj(context_embeddings).unsqueeze(1)

        # Transformer decoder
        # We let the memory be repeated if needed, or used as is
        out = self.transformer(
            tgt=token_emb,         # (batch, seq_len, hidden_dim)
            memory=memory          # (batch, 1, hidden_dim)
        )  # (batch, seq_len, hidden_dim)

        logits = self.output_proj(out)  # (batch, seq_len, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)
        preds = torch.argmax(logits, dim=-1)
        mask = (target_ids != self.pad_idx)
        correct = ((preds == target_ids) & mask).sum()
        acc = correct / mask.sum()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)
        preds = torch.argmax(logits, dim=-1)
        mask = (target_ids != self.pad_idx)
        acc = ((preds == target_ids) & mask).sum() / mask.sum()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        context_embeddings, input_ids, target_ids = batch
        logits = self(context_embeddings, input_ids)
        loss = self.criterion(logits.transpose(1, 2), target_ids)
        preds = torch.argmax(logits, dim=-1)
        mask = (target_ids != self.pad_idx)
        acc = ((preds == target_ids) & mask).sum() / mask.sum()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
