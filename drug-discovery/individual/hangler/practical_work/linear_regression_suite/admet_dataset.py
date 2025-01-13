import pandas as pd
import pickle
from coati.common.s3 import download_from_s3
import os

#### List of all Datasets ####

# 'ames': ('AUROC', 'Classification'),
# 'bace_regression': ('RMSE', 'Regression'),
# 'bace_classification': ('AUROC', 'Classification'),
# 'pgp_broccatelli': ('AUROC', 'Classification'),
# 'caco2_wang': ('RMSE', 'Regression'),
# 'clearance_microsome_az': ('Spearman', 'Regression'),
# 'clearance_hepatocyte_az': ('Spearman', 'Regression'),
# 'ppbr_az': ('RMSE', 'Regression'),
# 'vdss_lombardo': ('Spearman', 'Regression'),
# 'hia_hou': ('AUROC', 'Classification'),
# 'ld50_zhu': ('RMSE', 'Regression'),
# 'delaney': ('RMSE', 'Regression'),
# 'cyp3a4_veith': ('AUROC', 'Classification'),
# 'half_life_obach': ('Spearman', 'Regression'),
# 'bioavailability_ma': ('AUROC', 'Classification'),
# 'clintox': ('AUROC', 'Classification'),
# 'herg_karim': ('AUROC', 'Classification'),
# 'hiv': ('AUROC', 'Classification'),
# 'dili': ('AUROC', 'Classification'),
# 'cyp2c9_veith': ('AUROC', 'Classification'),
# 'herg': ('AUROC', 'Classification'),

dataset_info = {
        'ames': ('AUROC', 'Classification'),
        'bace_regression': ('RMSE', 'Regression'),
        'bace_classification': ('AUROC', 'Classification'),
        'pgp_broccatelli': ('AUROC', 'Classification'),
        'caco2_wang': ('RMSE', 'Regression'),
        'clearance_microsome_az': ('Spearman', 'Regression'),
        'clearance_hepatocyte_az': ('Spearman', 'Regression'),
        'ppbr_az': ('RMSE', 'Regression'),
        'vdss_lombardo': ('Spearman', 'Regression'),
        'hia_hou': ('AUROC', 'Classification'),
        'ld50_zhu': ('RMSE', 'Regression'),
        'delaney': ('RMSE', 'Regression'),
        'cyp3a4_veith': ('AUROC', 'Classification'),
        'half_life_obach': ('Spearman', 'Regression'),
        'bioavailability_ma': ('AUROC', 'Classification'),
        'clintox': ('AUROC', 'Classification'),
        'herg_karim': ('AUROC', 'Classification'),
        'hiv': ('AUROC', 'Classification'),
        'dili': ('AUROC', 'Classification'),
        'cyp2c9_veith': ('AUROC', 'Classification'),
        'herg': ('AUROC', 'Classification'),
}

def _load_dataset_links(file_path: str) -> list:
    """
    Load dataset links from a text file.

    Args:
        file_path (str): Path to the text file containing dataset links.

    Returns:
        list: A list of dataset URLs.
    """
    with open(file_path, 'r') as file:
        links = file.read().splitlines()
        
    return links

def download_admet_terray_data():
    """
    Download ADMET dataset from Terray's public S3 bucket.
    """
    links_path = 'admet_datasets.txt'
    dataset_links = _load_dataset_links(links_path)
    local_dir = './datasets'

    for link in dataset_links:
        # extract dataset_name from the link
        dataset_name = link.split('/')[-1]
        dataset_path = os.path.join(local_dir, dataset_name)

        # check if the dataset is already downloaded
        if os.path.exists(dataset_path):
            print(f"{dataset_name} already exists")
            continue
        
        try:
            download_from_s3(link)
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
            continue

def load_dataset(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data = pd.DataFrame(data)
    return data

def add_columns(data: pd.DataFrame, metric: str, task: str) -> pd.DataFrame:
    data['Metric'] = metric
    data['Task'] = task
    return data

def save_dataset(data: pd.DataFrame, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def update_datasets(base_path: str = 'datasets/'):
    for dataset_name, (metric, task) in dataset_info.items():
        file_path = f'{base_path}{dataset_name}.pkl'
        data = load_dataset(file_path)
        updated_data = add_columns(data, metric, task)
        save_dataset(updated_data, file_path)
        print(f"Updated {dataset_name} dataset")