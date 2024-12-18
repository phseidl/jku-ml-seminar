import numpy as np
import os
import torch
import pandas as pd
from math import exp
from rdkit import Chem
from rdkit.Chem import AllChem
#from fcd_torch.fcd_torch.fcd import FCD
from fcd import get_fcd, canonical_smiles, load_ref_model
from rdkit.Chem import rdmolfiles, rdmolops
from typing import List

import logging
from rdkit import RDLogger

# Suppress RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

# Alternatively, you can suppress warnings globally for RDKit
RDLogger.DisableLog('rdApp.*')  # Suppress all RDKit logging

def get_device() -> torch.device:
    """Determine available device.

    Returns:
        device(type='cuda') if cuda is available, device(type='cpu') otherwise.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def calculate_validity(smiles_list):
    """ Check if SMILES are valid by attempting to create RDKit molecules. """
    valid = [Chem.MolFromSmiles(sm) is not None for sm in smiles_list]
    validity = np.mean(valid)
    return validity

def calculate_novelty(generated_smiles, training_smiles):
    """ Calculate novelty as the proportion of generated SMILES not in the training set. """
    if training_smiles is None:
        return np.nan
    
    training_set = set(training_smiles)
    novel = [sm not in training_set for sm in generated_smiles]
    novelty = np.mean(novel)
    return novelty

def calculate_uniqueness(smiles_list):
    """ Calculate uniqueness as the proportion of unique SMILES in the generated list. """
    unique_smiles = len(set(smiles_list))
    uniqueness = unique_smiles / len(smiles_list)
    return uniqueness

def calculate_fcd(generated_smiles: str, reference_smiles: str, canonicalize: bool = True, n_jobs: int = 8) -> float:
    """ Calculate the FCD between generated molecules and a reference set. """

    model = load_ref_model()

    print(generated_smiles[:5])
    print(reference_smiles[:5])

    can_gen = canonical_smiles(s for s in canonical_smiles(generated_smiles) if s is not None)
    can_ref = canonical_smiles(s for s in canonical_smiles(reference_smiles) if s is not None)

    print("after canonicalization")
    print(can_gen[:5])
    print(can_ref[:5])

    fcd_score = get_fcd(can_ref, can_gen, model)

    # fcd_calculator = FCD(canonize=canonicalize, device=device, n_jobs=n_jobs)
    # fcd_score = fcd_calculator(reference_smiles, generated_smiles)

    return fcd_score

def evaluate_guacamol(
    generated_smiles: List, 
    training_smiles: List = None, 
    reference_dataset: List = None, 
    model_name: str = 'COATI', 
    results_path: str = 'guacamol_results.csv',
    ):
    """ Evaluate the generated SMILES against the training set or reference dataset. """

    # Calculate metrics
    validity = calculate_validity(generated_smiles)
    novelty = calculate_novelty(generated_smiles, training_smiles)
    uniqueness = calculate_uniqueness(generated_smiles)

    fcd_score = None
    if reference_dataset is not None:
        fcd_score = calculate_fcd(generated_smiles, reference_dataset)

    elif training_smiles is not None:
        fcd_score = calculate_fcd(generated_smiles, training_smiles)
    else:
        print("Either training_smiles_path or reference_dataset_path must be provided to calculate FCD.")


    # Print results
    print("Validity:", validity)
    print("Novelty:", novelty)
    print("Uniqueness:", uniqueness)
    print("FCD Score:", fcd_score if fcd_score is not None else np.nan)
    print("FCD GuacaMol:", exp(-0.2 * fcd_score) if fcd_score is not None else np.nan)

    # Store resutls into CSV (add if file exists)
    results = pd.DataFrame({
        "Model": [model_name],
        "Validity": [validity],
        "Novelty": [novelty],
        "Uniqueness": [uniqueness],
        "FCD Score": [fcd_score] if fcd_score is not None else np.nan,
        "FCD GuacaMol": [exp(-0.2 * fcd_score)] if fcd_score is not None else np.nan,
    })

    if os.path.exists(results_path):
        results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results.to_csv(results_path, mode='w', header=True, index=False)

if __name__ == "__main__":

    path_list = {
        "guacamol model": "generated_smiles/guacamol_model_generated_mols.csv",
        "grande closed": "generated_smiles/grande_closed_generated_mols.csv",
        "coati2": "generated_smiles/coati2_generated_mols.csv",
        "autoreg only": "generated_smiles/autoreg_only_generated_mols.csv",
    }

    training_smiles_path = "data/train_valid_test_guacamol.csv"

    # load train_smiles as df
    train_df = pd.read_csv(training_smiles_path)
    training_smiles = train_df['smiles'].tolist()


    for model_name, path in path_list.items():
        with open(path, 'r') as f:
            generated_smiles = f.readlines()

        evaluate_guacamol(generated_smiles, training_smiles_path, model_name=model_name)