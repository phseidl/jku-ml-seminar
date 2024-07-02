from coati.models.io import load_e3gnn_smiles_clip_e2e
from coati.models.simple_coati2.io import load_coati2
from downstream_tasks.batch_processing import embed_for_linear_probing
from downstream_tasks.evaluation import perform_model_analysis

from typing import List, Dict, Any
import torch
from torch import nn
import clamp
import numpy as np
import pandas as pd


def load_models(device: torch.device):
    """
    Loads various machine learning models and returns them in a dictionary for easy access by name.

    Args:
    device (torch.device): The device (CPU or GPU) to load the models onto.

    Returns:
    dict: A dictionary containing all models and tokenizers, accessible by their names.
    """

    models = {}

    """ 
    Grande Close Model:
        - Loss = InfoNCE + AR (AR is the autoregressive entropy loss)
        - E(3)-GNN = 5*256
        - Transformer = 16\*16\*256
        - Latent Dim. = 256
        - url = s3://terray-public/models/grande_closed.pkl 
    """

    print("COATI Model: Grande Closed\n")

    models['coati_grande_encoder'], models['coati_grande_tokenizer'] = load_e3gnn_smiles_clip_e2e(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/grande_closed.pkl"
    )

    """ 
    Autoregressive only Model:
        - Loss = AR
        - E(3)-GNN = N/A
        - Transformer = 16\*16\*256
        - Latent Dim. = 256
        - url = s3://terray-public/models/autoreg_only.pkl 
    """

    print("\n\n\nCOATI Model: Autorreg Only\n")

    models['coati_autoreg_encoder'], models['coati_autoreg_tokenizer'] = load_e3gnn_smiles_clip_e2e(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/autoreg_only.pkl"
    )

    """ 
    COATI2 Model:
        - trained on ~2x more data
        - Loss = InfoNCE + AR
        - chiral-aware 3D GNN = 5*256? (code not available)
        - Transformer = 16\*16\*256
        - Latent Dim. = 512 (new!)
        - url = s3://terray-public/models/coati2_chiral_03-08-24.pkl 
    """
    
    print("\n\n\nCOATI2 Model\n")

    models['coati2_encoder'], models['coati2_tokenizer'] = load_coati2(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl"
    )

    """
    CLAMP Model:
        - Compound Encoder = Input:8192x4096, Hidden:4096x2048, Output:2048x768
        - Assay Encoder = Input:512x4096, Hidden:4096x2048, Output:2048x768
        - Latent Dim. = 768
    """

    print("\n\n\nCLAMP Model\n")

    models['clamp_model']= clamp.CLAMP(device='cpu')
    models['clamp_model'].eval()

    print(models['clamp_model'])

    return models

def execute_linear_probing(data: pd.DataFrame, model_details: Dict[str, Any], dataset_name: str, analysis_type: List[str]) -> None:
    """
    Processes a given model: Computes embeddings, performs logistic regression analysis, and saves results.

    Args:
        data (pd.DataFrame): The combined DataFrame containing all necessary data for processing.
        model_details (Dict[str, Any]): Dictionary containing model-related data like the model itself, tokenizer (if applicable), and model name.
        dataset_name (str): The name of the dataset currently being processed, used for file naming.
        analysis_type (List[str]): The type of analysis to be performed, e.g., logistic_regression or/and due.

    Returns:
        None: This function performs operations in-place and saves files to disk without returning any value.
    """

    data_records = data.to_dict('records')

    model_name, encoder, tokenizer = model_details['name'], model_details['encoder'], model_details.get('tokenizer')
    embeddings, labels, indices, failed_smiles = embed_for_linear_probing(
        data_records,
        model_name=model_name,
        encoder=encoder,
        tokenizer=tokenizer
    )

    # Save embeddings, indices, and labels
    np.save(f'{model_name}_{dataset_name}_embeddings.npy', embeddings)
    np.save(f'{model_name}_{dataset_name}_indices.npy', indices)
    labels_df = pd.DataFrame(labels, columns=['Label'])
    labels_df.to_csv(f'{model_name}_{dataset_name}_labels.csv', index=False)
    print(f"Saved embeddings, indices, and labels for {model_name} on {dataset_name}")

    # delete failed smiles in data
    data = data[~data['CanonicalSMILES'].isin(failed_smiles)]

    # Perform logistic regression for each assay and scaffold
    assays = data['assay_idx'].unique()
    for analysis in analysis_type:
        for assay in assays:
            assay_specific_data = data[data['assay_idx'] == assay]
            print(f"Processing assay {assay}:")
            # Perform logistic regression for default and multiple scaffold splits
            for i in range(11):  # Includes default (0-9) and multi-scaffold (not specified as 10 but logical extension)
                split_key = 'scaffold_split' if i == 0 else f'scaffold_split_{i-1}'
                train_idx = np.where(assay_specific_data[split_key] == 'train')[0]
                test_idx = np.where(assay_specific_data[split_key] == 'test')[0]

                perform_model_analysis(
                    embeddings, labels, train_idx, test_idx,
                    analysis_type=analysis,
                    dataset_name=f'{dataset_name}_{split_key}',
                    model_name=model_name,
                    assay_idx=assay,
                    results_file='linear_probing_results.csv'
                )