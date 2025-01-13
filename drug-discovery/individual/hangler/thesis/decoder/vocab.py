from typing import List, Dict

class Vocabulary:
    """Simple vocabulary class to handle token <-> index conversions."""
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}

        self.pad_token = "<pad>"
        self.start_token = "<sos>"
        self.end_token = "<eos>"

        # Ensure special tokens are in the vocabulary
        for special in [self.pad_token, self.start_token, self.end_token]:
            if special not in self.token_to_idx:
                self.token_to_idx[special] = len(self.token_to_idx)
                self.idx_to_token[len(self.idx_to_token)] = special

        self.pad_idx = self.token_to_idx[self.pad_token]
        self.start_idx = self.token_to_idx[self.start_token]
        self.end_idx = self.token_to_idx[self.end_token]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(t, self.pad_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token[i] for i in indices if i in self.idx_to_token]

    def __len__(self):
        return len(self.token_to_idx)
