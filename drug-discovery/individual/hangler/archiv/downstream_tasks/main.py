from downstream_tasks.dataset import download_and_extract_data, list_datasets
from downstream_tasks.data_utils import prepare_data
from downstream_tasks.model_utils import load_models, execute_linear_probing

import numpy as np
import os
import torch
import pandas as pd

def main():
    base_url = "https://cloud.ml.jku.at/s/pyJMm4yQeWFM2gG/download"
    data_directory = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Download and prepare datasets
    download_and_extract_data(base_url, data_directory)
    datasets = list_datasets(os.path.join(data_directory, 'downstream'))

    # Load models
    models = load_models(device)

    # Dictionary to map model configurations
    model_configurations = {
        'coati_grande': {'encoder': models['coati_grande_encoder'], 'tokenizer': models['coati_grande_tokenizer']},
        'coati_autoreg': {'encoder': models['coati_autoreg_encoder'], 'tokenizer': models['coati_autoreg_tokenizer']},
        'coati2': {'encoder': models['coati2_encoder'], 'tokenizer': models['coati2_tokenizer']},
        'clamp': {'encoder': models['clamp_model']}  # No tokenizer needed
    }

    # Process each dataset with each model configuration
    for dataset_name, activity_path, smiles_path in datasets:
        data = prepare_data(smiles_path, activity_path)

        for model_key, model_details in model_configurations.items():
            print(f"Processing {model_key} for dataset {dataset_name}")
            model_details['name'] = model_key 
            execute_linear_probing(data, model_details, dataset_name, ['logistic_regression', 'due'])

if __name__ == "__main__":
    # pip install -r requirements.txt

    main()