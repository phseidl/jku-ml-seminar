import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Subset
import torch

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
        val_ratio=0.1,
        max_dataset_size=None,
        max_smiles_length=None
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
        self.max_dataset_size = max_dataset_size
        self.max_smiles_length = max_smiles_length

    def setup(self, stage=None):
        full_dataset = ChunkedMoleculeDataset(
            embedding_file_paths=self.embedding_file_paths,
            tokenized_smiles_file=self.tokenized_smiles_path,
            vocab=self.vocab,
            max_length=self.max_length
        )  # emb_tensor, input_ids, target_ids

        # Filter out SMILES that exceed self.max_smiles_length
        if self.max_smiles_length is not None:
            indices_to_keep = []
            for idx in range(len(full_dataset)):
                raw_smiles_tokens = self._get_raw_smiles_tokens(full_dataset, idx)
                if len(raw_smiles_tokens) <= self.max_smiles_length:
                    indices_to_keep.append(idx)

            print(f"Filtered out {len(full_dataset) - len(indices_to_keep)} samples that exceed max SMILES length.")
            full_dataset = Subset(full_dataset, indices_to_keep)

        # Limit dataset size to max_dataset_size
        if self.max_dataset_size is not None:
            limit = min(self.max_dataset_size, len(full_dataset))
            subset_indices = list(range(limit))
            full_dataset = Subset(full_dataset, subset_indices)


        # Validation: Check consistency
        # print("Validating subset consistency...")
        # for i, subset_idx in enumerate(full_dataset.indices):
        #     original_data = full_dataset.dataset[subset_idx]
        #     subset_data = full_dataset[i]

        #     # Compare embeddings, input_ids, and target_ids for consistency
        #     if not torch.equal(original_data[0], subset_data[0]) or \
        #     not torch.equal(original_data[1], subset_data[1]) or \
        #     not torch.equal(original_data[2], subset_data[2]):
        #         print(f"Data mismatch at index {i}: subset index {subset_idx}")
        #         print(f"Original data: {original_data}")
        #         print(f"Subset data: {subset_data}")
        #         raise ValueError("Subset indices or data alignment is incorrect!")

        # print("Subset indices and data alignment are correct.")

        # Split dataset into train, val, and test
        n = len(full_dataset)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )


    def _get_raw_smiles_tokens(self, dataset, idx):
        if isinstance(dataset, Subset):
            actual_dataset = dataset.dataset
            real_idx = dataset.indices[idx]
        else:
            actual_dataset = dataset
            real_idx = idx

        tokens = actual_dataset.tokenized_smiles[real_idx]
        return tokens

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == "__main__":
    from decoder.vocab import Vocabulary
    import logging as logger

    # Setup logging
    logger.basicConfig(level=logger.INFO)
    logger.info("Starting MoleculeDataModule validation...")

    # Load the vocabulary
    vocab_file_path = "decoder/vocab.txt"
    with open(vocab_file_path, "r", encoding="utf-8") as vf:
        tokens = [line.strip() for line in vf if line.strip()]
    vocab = Vocabulary(tokens)

    # Prepare chunked file lists
    embedding_file_paths = [f"data/single embeddings/embeddings_pubchem10m.npy_{i}.npy" for i in range(100)]
    tokenized_smiles_path = "data/tokenized_smiles.npy"

    # Define DataModule parameters
    # max_length = 152  # Maximum length for tokenized SMILES (+<sos>, <eos>, and padding)
    batch_size = 128
    max_dataset_size = 10000  # Limit the dataset size for quicker validation
    max_smiles_length = 90  # Filter out overly long SMILES strings

    # 4) Initialize the DataModule
    dm = MoleculeDataModule(
        vocab=vocab,
        embedding_file_paths=embedding_file_paths,
        tokenized_smiles_path=tokenized_smiles_path,
        batch_size=batch_size,
        num_workers=4,
        max_dataset_size=max_dataset_size,
        max_smiles_length=max_smiles_length
    )

    # 5) Setup the DataModule
    logger.info("Setting up the data module...")
    dm.setup()

    # 6) Validate the data loaders
    logger.info("Validating train dataloader...")
    for batch in dm.train_dataloader():
        emb, input_ids, target_ids = batch
        logger.info(f"Train batch: emb={emb.shape}, input_ids={input_ids.shape}, target_ids={target_ids.shape}")
        break  # Only check the first batch

    logger.info("Validating val dataloader...")
    for batch in dm.val_dataloader():
        emb, input_ids, target_ids = batch
        logger.info(f"Val batch: emb={emb.shape}, input_ids={input_ids.shape}, target_ids={target_ids.shape}")
        break  # Only check the first batch

    logger.info("Validating test dataloader...")
    for batch in dm.test_dataloader():
        emb, input_ids, target_ids = batch
        logger.info(f"Test batch: emb={emb.shape}, input_ids={input_ids.shape}, target_ids={target_ids.shape}")
        break  # Only check the first batch

    logger.info("MoleculeDataModule validation complete.")