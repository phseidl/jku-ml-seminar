import os
import torch
import logging as logger
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
#from pytorch_lightning.profiler import PyTorchProfiler


from decoder.vocab import Vocabulary
from decoder.data_module import MoleculeDataModule
from decoder.decoder_model import LSTMDecoder

if __name__ == "__main__":
    # Setup logging
    logger.basicConfig(level=logger.INFO)
    logger.info("Starting training process...")

    torch.set_float32_matmul_precision('medium')

    # 1) Load the tokens from vocab.txt
    vocab_file_path = "decoder/vocab.txt"
    with open(vocab_file_path, "r", encoding="utf-8") as vf:
        tokens = [line.strip() for line in vf if line.strip()]
    vocab = Vocabulary(tokens)

    # 2) Prepare chunked file lists
    # For example: 100 .npy files
    # /Volumes/PHILIPS/thesis/data/embeddings_pubchem10m/single embeddings/embeddings_pubchem10m.npy_0.npy
    embedding_file_paths = [f"data/single embeddings/embeddings_pubchem10m.npy_{i}.npy" for i in range(100)]
    # Similarly for tokenized SMILES if you also splitted them into 100 files
    tokenized_smiles_path = "data/tokenized_smiles.npy"

    # ---------------------------
    # Create DataModule
    max_length = 152 # maximum length of tokenized SMILES + 3 for <sos>, <eos>, and padding
    batch_size = 64

    dm = MoleculeDataModule(
        vocab=vocab,
        embedding_file_paths=embedding_file_paths,
        tokenized_smiles_path=tokenized_smiles_path,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=8  # tune this
    )

    dm.setup()

    # ---------------------------
    # Create Model
    embedding_dim = 768 # latent dim of encoder/embeddings
    model = LSTMDecoder(
        vocab_size=len(vocab), 
        embedding_dim=embedding_dim, 
        hidden_dim=768,
        decoder_layers=2,
        lr=1e-3,
        pad_idx=vocab.pad_idx
    )

    pl_logger = CSVLogger(save_dir="lightning_logs/")

    # ---------------------------
    # Checkpoints and Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="decoder-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=50,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto" if torch.cuda.device_count() <= 1 else DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        precision="16-mixed",  # Enable mixed precision for faster training
        accumulate_grad_batches=4,
        logger=pl_logger,
        log_every_n_steps=1000
    )

    # ---------------------------
    # Training
    trainer.fit(model, dm)

    # ---------------------------
    # Testing
    trainer.test(model, datamodule=dm)

    logger.info("Training and testing complete.")
