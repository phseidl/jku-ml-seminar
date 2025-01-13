import torch
import pytorch_lightning as pl
from torch import nn
from xbert import BertConfig, BertForMaskedLM

class XbertDecoder(pl.LightningModule):
    def __init__(self, config_path: str, pad_idx=0, lr=2e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx = pad_idx
        self.lr = lr
        self.weight_decay = weight_decay

        # Load config and instantiate model
        bert_config = BertConfig.from_json_file(config_path)
        bert_config.add_cross_attention = True  # ensure cross-attention is enabled
        self.model = BertForMaskedLM(config=bert_config)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, context_embeddings, input_ids):
        """
        context_embeddings: (batch, embed_dim)
        input_ids: (batch, seq_len)
        """
        # Expand context to (batch, seq_len, embed_dim) for cross-attention
        batch_size, seq_len = input_ids.size()
        # Treat context_embeddings as "encoder_hidden_states"
        # We mask out attention to keep dimension alignment
        encoder_hidden_states = context_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        outputs = self.model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        logits = outputs.logits  # (batch, seq_len, vocab_size)
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
