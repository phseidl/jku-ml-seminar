import logging as logger
import re
from typing import List
from collections import Counter
from tqdm import tqdm

from decoder.vocab import Vocabulary
from smiles_tokenizer import segment_smiles, validate_smiles

logger.basicConfig(level=logger.INFO)

if __name__ == "__main__":
    logger.info("Loading SMILES data from text file and checking validity with RDKit...")

    smiles_file = "data/pubchem_10m.txt"  # Path to your SMILES text file
    try:
        with open(smiles_file, "r", encoding="utf-8") as f:
            smiles_lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"File {smiles_file} not found. Please add your SMILES file.")
        exit(1)

    smiles_list = []
    bracket_counts = Counter()

    # Read lines, validate SMILES, and tokenize
    for smi in tqdm(smiles_lines, desc="Processing SMILES", unit="smiles"):
        smi = smi.strip()
        if not smi:
            continue  # Skip empty lines

        # Check if SMILES is valid using RDKit
        mol = validate_smiles(smi)
        if mol is None:
            # Not a valid molecule
            logger.warning(f"Invalid SMILES (RDKit parsing failed): {smi}")
            continue

        # Tokenize the valid SMILES
        tokens = segment_smiles(smi)
        smiles_list.append(tokens)

        # keep track of bracketed expressions
        bracket_expressions = re.findall(r"\[[^\]]+\]", smi)
        for expr in bracket_expressions:
            bracket_counts[expr] += 1

    # Print the most common bracketed expressions
    if bracket_counts:
        logger.info("Top 10 bracketed expressions in dataset:")
        for expr, freq in bracket_counts.most_common(10):
            logger.info(f"  {expr}: {freq} occurrences")

    # Build a set of all tokens from the valid SMILES
    all_tokens = set(token for smi_tokens in smiles_list for token in smi_tokens)
    token_list = sorted(list(all_tokens))

    logger.info(f"Number of unique tokens: {len(token_list)}")
    if token_list:
        logger.info(f"Example tokens: {token_list[:50]}")

    # Create the vocabulary
    vocab = Vocabulary(token_list)
    logger.info("Vocabulary successfully created.")
    logger.info(f"Vocabulary size (including special tokens): {len(vocab)}")

    # Save the vocabulary to a file
    vocab_file = "decoder/vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as vf:
        for token in tqdm(vocab.tokens, desc="Writing Vocabulary", unit="token"):
            vf.write(token + "\n")

    logger.info(f"Vocabulary written to {vocab_file}")
