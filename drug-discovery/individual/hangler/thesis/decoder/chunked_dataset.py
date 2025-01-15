import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List

class ChunkedMoleculeDataset(Dataset):
    """
    A dataset that handles multiple .npy embedding files and 
    a single .npy file for tokenized SMILES entries.
    """
    def __init__(
        self,
        embedding_file_paths: List[str],
        tokenized_smiles_file: str,
        vocab,
        max_length: int
    ):
        """
        Args:
            embedding_file_paths: list of paths to .npy files, e.g. embeddings_00.npy, ...
            tokenized_smiles_file: path to a single .npy file for tokenized SMILES.
            vocab: the Vocabulary object
            max_length: maximum sequence length
        """
        super().__init__()
        self.vocab = vocab
        self.max_length = max_length
        
        # Memory-map each embeddings file
        self.embedding_file_paths = embedding_file_paths
        self.embeddings_memmaps = []
        self.sizes = []  # how many samples in each file
        
        for path in self.embedding_file_paths:
            #print(f"Memory-mapping: {path}...")
            arr = np.load(path, mmap_mode="r")
            self.embeddings_memmaps.append(arr)
            self.sizes.append(arr.shape[0])
        
        # Build a prefix sum of sizes to know indexing boundaries
        self.offsets = [0]
        for sz in self.sizes:
            self.offsets.append(self.offsets[-1] + sz)
        
        # Load the single tokenized SMILES file into memory
        self.tokenized_smiles = np.load(tokenized_smiles_file, allow_pickle=True)

        # Ensure consistency
        assert self.__len__() == len(self.tokenized_smiles), (
            "Mismatch between total number of embeddings and tokenized SMILES."
        )

    def __len__(self):
        # Total samples = sum of sizes
        return self.offsets[-1]

    def get_raw_smiles_tokens(self, idx):
        """
        Return the raw tokenized SMILES (as loaded from self.tokenized_smiles)
        for the provided 'idx', WITHOUT adding <sos>, <eos>, or padding.
        This helps the MoleculeDataModule check the SMILES length easily.
        """
        return self.tokenized_smiles[idx]

    
    def __getitem__(self, idx):
        # Find which file chunk 'idx' falls into
        file_idx = None
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= idx < self.offsets[i+1]:
                file_idx = i
                break
        
        # 2) Local index in that file
        local_idx = idx - self.offsets[file_idx]

        # 3) Load the embedding row from memory-map
        emb = self.embeddings_memmaps[file_idx][local_idx]

        # 4) Load the raw tokenized SMILES from self.tokenized_smiles
        #    Then add <sos>, <eos>, etc.
        tokens = self.tokenized_smiles[idx]
        tokens = [self.vocab.start_token] + list(tokens) + [self.vocab.end_token]
        
        if len(tokens) < self.max_length:
            tokens += [self.vocab.pad_token] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Prepare input_tokens vs target_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        # Convert to tensors
        input_ids = torch.tensor(self.vocab.encode(input_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.vocab.encode(target_tokens), dtype=torch.long)
        emb_tensor = torch.tensor(emb, dtype=torch.float)

        return emb_tensor, input_ids, target_ids
