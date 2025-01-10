# chunked_data_module.py

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from chunked_dataset import ChunkedMoleculeDataset

class MoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocab,
        embedding_file_paths,
        tokenized_smiles_path,
        max_length=152,
        batch_size=64,
        num_workers=4,
        train_ratio=0.9,
        val_ratio=0.1
    ):
        super().__init__()
        self.vocab = vocab
        self.embedding_file_paths = embedding_file_paths
        self.tokenized_smiles_path = tokenized_smiles_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

    def setup(self, stage=None):
        full_dataset = ChunkedMoleculeDataset(
            embedding_file_paths=self.embedding_file_paths,
            tokenized_smiles_file=self.tokenized_smiles_path,
            vocab=self.vocab,
            max_length=self.max_length
        )
        n = len(full_dataset)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
