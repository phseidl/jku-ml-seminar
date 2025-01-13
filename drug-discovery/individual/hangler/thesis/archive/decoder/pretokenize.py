import argparse
import numpy as np
import pickle
from tqdm import tqdm
import logging as logger
from smiles_tokenizer import segment_smiles, validate_smiles

# Configure logging
logger.basicConfig(level=logger.INFO)

def pretokenize_smiles(input_file: str, output_file: str, format: str = "npy"):
    """
    Pre-tokenize SMILES strings from an input text file and save the tokenized result in a specified format.
    Args:
        input_file: Path to the input text file containing one SMILES string per line.
        output_file: Path to save the preprocessed file.
        format: The format to save the tokenized SMILES ('npy' or 'pkl').
    """
    logger.info(f"Reading SMILES from {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"File {input_file} not found.")
        return

    logger.info(f"Tokenizing and validating {len(smiles_list)} SMILES strings...")
    tokenized_smiles = []
    valid_count = 0

    for smiles in tqdm(smiles_list, desc="Processing SMILES", unit="smiles"):
        # Use segment_smiles directly in the tokenization logic
        if validate_smiles(smiles):
            tokens = segment_smiles(smiles)
            tokenized_smiles.append(tokens)
            valid_count += 1
        else:
            logger.warning(f"Invalid SMILES: {smiles}")

    logger.info(f"Finished processing. Valid SMILES: {valid_count}/{len(smiles_list)}")

    # Save the tokenized SMILES to the specified file
    if format == "npy":
        logger.info(f"Saving tokenized SMILES to {output_file} in .npy format...")
        np.save(output_file, np.array(tokenized_smiles, dtype=object))
    elif format == "pkl":
        logger.info(f"Saving tokenized SMILES to {output_file} in .pkl format...")
        with open(output_file, "wb") as f:
            pickle.dump(tokenized_smiles, f)
    else:
        logger.error(f"Unsupported format: {format}. Use 'npy' or 'pkl'.")
        return

    logger.info(f"Tokenized SMILES saved successfully to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize SMILES strings and save them to a file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input SMILES text file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the tokenized SMILES.")
    parser.add_argument("--format", type=str, choices=["npy", "pkl"], default="npy", help="Output file format.")
    args = parser.parse_args()

    # Call the method with parsed arguments
    pretokenize_smiles(args.input, args.output, args.format)