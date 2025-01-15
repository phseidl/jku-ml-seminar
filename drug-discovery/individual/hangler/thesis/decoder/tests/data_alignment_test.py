import numpy as np
from decoder.chunked_dataset import ChunkedMoleculeDataset
from decoder.vocab import Vocabulary
from tqdm import tqdm

def check_data_alignment(embedding_file_paths, tokenized_smiles_path, vocab, max_length):
    """
    Check if embeddings are correctly aligned with their corresponding tokenized SMILES.
    
    Args:
        embedding_file_paths: List of paths to the embedding .npy files.
        tokenized_smiles_path: Path to the tokenized SMILES .npy file.
        vocab: The Vocabulary object.
        max_length: The maximum length of tokenized SMILES.
    """
    # Load all tokenized SMILES
    tokenized_smiles = np.load(tokenized_smiles_path, allow_pickle=True)

    max_length = max(len(smile) for smile in tokenized_smiles) + 3 # Add 3 for <sos>, <eos>, and padding
    print(f"The maximum length of pre-tokenized SMILES is: {max_length}")

    # Create the dataset 
    dataset = ChunkedMoleculeDataset(
        embedding_file_paths=embedding_file_paths,
        tokenized_smiles_file=tokenized_smiles_path,
        vocab=vocab,
        max_length=max_length
    )

    # Iterate through the dataset and check alignment
    for idx in tqdm(range(len(dataset)), desc="Checking data alignment"):
        # Get data from dataset
        embedding, input_ids, target_ids = dataset[idx]

        # Check alignment with tokenized SMILES
        original_smiles_tokens = tokenized_smiles[idx]
        input_tokens = vocab.decode(input_ids.numpy())
        target_tokens = vocab.decode(target_ids.numpy())

        # Remove <sos>, <eos>, and padding for a clean comparison
        input_tokens_cleaned = [token for token in input_tokens if token not in {vocab.start_token, vocab.end_token, vocab.pad_token}]
        target_tokens_cleaned = [token for token in target_tokens if token not in {vocab.start_token, vocab.end_token, vocab.pad_token}]

        # Compare original tokens with input/target tokens
        if input_tokens_cleaned != original_smiles_tokens or target_tokens_cleaned != original_smiles_tokens:
            print(f"Data mismatch at index {idx}:")
            print(f"Original SMILES Tokens: {original_smiles_tokens}")
            print(f"Input Tokens: {input_tokens_cleaned}")
            print(f"Target Tokens: {target_tokens_cleaned}")
            return False

    return True

def check_embedding_alignment(embedding_file_paths, tokenized_smiles_path, vocab, max_length):
    """
    Check if embeddings returned by the dataset are aligned with the embeddings 
    directly loaded from the .npy files at the same index.

    Args:
        embedding_file_paths: List of paths to the embedding .npy files.
        tokenized_smiles_path: Path to the tokenized SMILES .npy file.
        vocab: The Vocabulary object.
        max_length: The maximum length of tokenized SMILES.
    """
    # Load all tokenized SMILES
    tokenized_smiles = np.load(tokenized_smiles_path, allow_pickle=True)

    # Initialize dataset
    dataset = ChunkedMoleculeDataset(
        embedding_file_paths=embedding_file_paths,
        tokenized_smiles_file=tokenized_smiles_path,
        vocab=vocab,
        max_length=max_length
    )

    # Load embeddings directly from .npy files for comparison
    loaded_embeddings = []
    for path in embedding_file_paths[:3]:  # Limit to the first 3 files for testing
        loaded_embeddings.append(np.load(path, mmap_mode="r"))

    # Flatten the list of embeddings to access by global index
    flattened_embeddings = np.concatenate(loaded_embeddings, axis=0)

    # Iterate through the dataset and validate alignment
    for idx in tqdm(range(len(flattened_embeddings)), desc="Validating embedding alignment"):
        # Get embedding from the dataset
        dataset_embedding, _, _ = dataset[idx]

        # Get the embedding from the flattened list
        direct_embedding = flattened_embeddings[idx]

        # Check if embeddings match
        if not np.array_equal(dataset_embedding.numpy(), direct_embedding):
            print(f"Embedding mismatch at index {idx}:")
            print(f"Dataset embedding: {dataset_embedding.numpy()}")
            print(f"Direct embedding:  {direct_embedding}")
            return False

    print("All embeddings are correctly aligned with the dataset.")
    return True


if __name__ == "__main__":
    # Example configuration
    embedding_file_paths = [f"data/single embeddings/embeddings_pubchem10m.npy_{i}.npy" for i in range(100)]
    tokenized_smiles_path = "data/tokenized_smiles.npy"
    vocab_file_path = "decoder/vocab.txt"

    #smiles_test = ['C', '=', 'C', '1', 'C', 'C', '2', 'C', '=', 'N', 'c', '3', 'c', 'c', '(', 'O', 'C', 'C', 'C', 'C', 'C', 'O', 'c', '4', 'c', 'c', '5', 'c', '(', 'c', 'c', '4', 'O', 'C', ')', 'C', '(', '=', 'O', ')', 'N', '4', 'C', 'C', '(', '=', 'C', ')', 'C', 'C', '4', 'C', '(', 'O', ')', 'N', '5', 'C', '(', '=', 'O', ')', 'O', 'C', 'c', '4', 'c', 'c', 'c', '(', 'N', 'C', '(', '=', 'O', ')', 'C', '(', 'C', ')', 'N', 'C', '(', '=', 'O', ')', 'C', '(', 'N', 'C', '(', '=', 'O', ')', 'C', 'C', 'O', 'C', 'C', 'N', 'C', '(', '=', 'O', ')', 'C', 'C', 'N', '5', 'C', '(', '=', 'O', ')', 'C', '=', 'C', 'C', '5', '=', 'O', ')', 'C', '(', 'C', ')', 'C', ')', 'c', 'c', '4', ')', 'c', '(', 'O', 'C', ')', 'c', 'c', '3', 'C', '(', '=', 'O', ')', 'N', '2', 'C', '1']
    #print("len: ", len(smiles_test))

    # Load vocabulary
    with open(vocab_file_path, "r", encoding="utf-8") as vf:
        tokens = [line.strip() for line in vf if line.strip()]
    vocab = Vocabulary(tokens)

    # Maximum sequence length
    max_length = 150

    # Run the alignment check
    # is_aligned = check_data_alignment(embedding_file_paths, tokenized_smiles_path, vocab, max_length)
    # if not is_aligned:
    #     print("Data alignment failed. Please check your preprocessing or data loading.")
    # else:
    #     print("Data alignment check passed.")

    is_aligned = check_embedding_alignment(embedding_file_paths, tokenized_smiles_path, vocab, max_length)
    if not is_aligned:
        print("Embedding alignment failed. Please check your dataset or indexing logic.")
    else:
        print("Embedding alignment check passed.")
