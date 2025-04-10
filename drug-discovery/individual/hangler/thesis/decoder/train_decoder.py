import os
import torch
import logging as logger
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
#from pytorch_lightning.profiler import PyTorchProfiler

from decoder.vocab import Vocabulary
from decoder.data_module import MoleculeDataModule
from decoder.decoder_xbert import XbertDecoder

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
    tokenized_smiles_path = "data/tokenized_smiles.npy"

    # ---------------------------
    # Create DataModule
    max_length = 152 # maximum length of tokenized SMILES + 3 for <sos>, <eos>, and padding
    batch_size = 256 # 128

    dm = MoleculeDataModule(
        vocab=vocab,
        embedding_file_paths=embedding_file_paths,
        tokenized_smiles_path=tokenized_smiles_path,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=8,  # tune this
        max_dataset_size=100000
    ) # emb_tensor, input_ids, target_ids

    dm.setup()

    # ---------------------------
    # Create Model
    pretrain_config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 128,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_decoder': './config_decoder_clamp.json',
        'schedular': {'sched': 'cosine', 'lr': 0.5e-4, 'epochs': 5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 0.5e-4, 'weight_decay': 0.02}
    }

    embedding_dim = 768 # latent dim of encoder/embeddings

    model = XbertDecoder(
        config=pretrain_config,
        pad_idx=vocab.pad_idx,
        embed_dim=embedding_dim, 
        use_linear=False,  # True if you need dimension adaptation
        vocab = vocab
    )

    # ---------------------------
    # Checkpoints and Trainer
    checkpoint_dir = "checkpoints/"
    pl_logger = CSVLogger(save_dir="lightning_logs/")

    # Custom Checkpoint Callback to Save Every 20 Steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="decoder-{step:06d}-{val_loss:.4f}",
        save_top_k=-1,  # Save all checkpoints
        every_n_train_steps=50,  # Save every 20 training steps
        monitor=None,  # No validation metric monitoring for step-based saving
    )

    # Learning Rate Monitor Callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Resume from Last Checkpoint
    if os.path.exists(checkpoint_dir):
        last_checkpoint = None
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if checkpoints:
            last_checkpoint = max(
                [os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints],
                key=os.path.getctime
            )
            print(f"Resuming from the latest checkpoint: {last_checkpoint}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    else:
        last_checkpoint = None
        print("Checkpoint directory does not exist. Starting training from scratch.")

    # Trainer with Step-Based Validation and Checkpoint Intervals
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=5,  # Total number of epochs
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto" if torch.cuda.device_count() <= 1 else "ddp",
        callbacks=[
            checkpoint_callback,
            lr_monitor,
        ],
        precision="16-mixed",  # Enable mixed precision for faster training
        logger=pl_logger,
        val_check_interval=50,  # Validate every 20 steps
    )

    # ---------------------------
    # Training
    if last_checkpoint:
        trainer.fit(model, dm, ckpt_path=last_checkpoint)
    else:
        trainer.fit(model, dm)

    # ---------------------------
    # Testing
    trainer.test(model, datamodule=dm)

    logger.info("Training and testing complete.")
