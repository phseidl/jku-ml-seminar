import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
from scipy import stats
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import os

from linear_regression_suite.admet_dataset import load_dataset, dataset_info

def bootstrap_metric(function, n: int=500):
    "wrapper for metrics to bootstrap e.g. calc std"
    def wrapper(y_true, y_pred, sample_weight=None):
        l = (len(y_true))
        res = []
        for i in range(n):
            s = np.random.choice(range(l), l, replace=True)
            if not len(np.unique(y_true[s]))==2:
                continue
            else:
                res.append(function(y_true[s], y_pred[s], sample_weight=None if sample_weight is None else sample_weight[s]))#,
        return np.array(res)
    return wrapper

def bootstrap_metric_rmse(function, n: int=500):
    "Wrapper for metrics to bootstrap e.g. calc std"
    def wrapper(y_true, y_pred, sample_weight=None):
        l = len(y_true)
        res = []
        for i in range(n):
            s = np.random.choice(range(l), l, replace=True)
            if len(np.unique(y_true[s])) > 1:  # Ensure there is more than one unique value
                res.append(function(y_true[s], y_pred[s]))
        if len(res) == 0:  # Handle case where no valid resamples are found
            return np.array([np.nan])
        return np.array(res)
    return wrapper


def bootstrap_metric_spearman(function, n: int=500):
    "Wrapper for metrics to bootstrap e.g. calc std"
    def wrapper(y_true, y_pred, sample_weight=None):
        l = len(y_true)
        res = []
        for i in range(n):
            s = np.random.choice(range(l), l, replace=True)
            if len(np.unique(y_true[s])) > 1:
                res.append(function(y_true[s], y_pred[s])[0])  # Only the correlation coefficient is needed
        if len(res) == 0:  # Handle case where no valid resamples are found
            return np.array([np.nan])
        return np.array(res)
    return wrapper

def davgp_score(y_true, y_pred, sample_weight=None):
    avgp = average_precision_score(y_true, y_pred, sample_weight=sample_weight)
    y_avg = np.average(y_true, weights=sample_weight)
    return avgp - y_avg

def scaffold_split(dataframe, smiles_col='smiles', seed=None):
    np.random.seed(seed)
    scaffolds = {}
    for index, row in dataframe.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [index]
        else:
            scaffolds[scaffold].append(index)

    # Shuffle scaffold order for random splitting
    scaffold_items = list(scaffolds.items())
    np.random.shuffle(scaffold_items)

    train_indices, test_indices = [], []
    for scaffold, indices in scaffold_items:
        if len(train_indices) < len(test_indices):
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return dataframe.loc[train_indices], dataframe.loc[test_indices]

def train_and_evaluate(dataset_name, data, embeddings, task, metric, results_file="coati_linear_probing.csv" ,seed=42):
    X_train = data[0][embeddings].values.tolist()
    X_test = data[1][embeddings].values.tolist()
    y_train = data[0]['target'].values
    y_test = data[1]['target'].values

    # Select model based on task
    if task == 'Classification':
        model = LogisticRegression(max_iter=1500, class_weight='balanced', C=1, random_state=seed)
    elif task == 'Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Task {task} not supported")
    
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    if metric == 'AUROC':
        AUROC_score = roc_auc_score(y_test, y_pred)
        roc_auc_std = bootstrap_metric(roc_auc_score, n=500)(y_test, y_pred).std()

        avgp = average_precision_score(y_test, y_pred)
        avgp_std = bootstrap_metric(average_precision_score, n=500)(y_test, y_pred).std()

        davgp = avgp - y_test.mean()
        davgp_std = bootstrap_metric(davgp_score, n=500)(y_test, y_pred).std()

        print(f"AUROC={AUROC_score:.3f}, dAP={davgp:.3f}")

    elif metric == 'RMSE':
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_std = np.nan        
        print(f"RMSE={rmse:.3f}, RMSE std={rmse_std:.3f}")
    elif metric == 'Spearman':
        spearman = stats.spearmanr(y_test, y_pred)[0]
        spearman_std = np.nan
        print(f"Spearman={spearman:.3f}, Spearman std={spearman_std:.3f}")
    else:
        raise ValueError(f"Metric {metric} not supported")

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Seed': [seed],
        'Analysis Type': [task],
        'Model': [embeddings],
        'AUROC Score': [AUROC_score] if metric == 'AUROC' else [np.nan],
        'AUROC std': [roc_auc_std] if metric == 'AUROC' else [np.nan],
        'RMSE': [rmse] if metric == 'RMSE' else [np.nan],
        'RMSE std': [rmse_std] if metric == 'RMSE' else [np.nan],
        'Spearman': [spearman] if metric == 'Spearman' else [np.nan],
        'Spearman std': [spearman_std] if metric == 'Spearman' else [np.nan],
        'avgp': [avgp] if metric == 'AUROC' else [np.nan],
        'avgp_std': [avgp_std] if metric == 'AUROC' else [np.nan],
        'davgp': [davgp] if metric == 'AUROC' else [np.nan],
        'davgp_std': [davgp_std] if metric == 'AUROC' else [np.nan]
    })

    # Save or append the results to a CSV file
    # Check if the file exists to decide whether to write headers or not
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, mode='w', header=True, index=False)


def coati_linear_probing_eval(base_path: str='/Users/stefanhangler/Documents/Uni/Msc_AI/3_Semester/Seminar_Practical Work/Code.nosync/COATI/linear_regression_suite/datasets/'):
    n_runs = 10

    for dataset_name, (metric, task) in dataset_info.items():
        print(f"Start Training {task} for {dataset_name}:")
        file_path = f'{base_path}{dataset_name}.pkl'
        data = load_dataset(file_path)

        model_embeddings = ['Autoreg Only']

        try:
            scores = []
            for seed in range(n_runs):  # Perform 10 different scaffold splits
                print(f"Seed: {seed}")
                for model in model_embeddings:
                    tmp_data = data.copy()
                    # delete rows if value is nan in any of the model_embeddings columns
                    tmp_data = tmp_data.dropna(subset=[model])

                    split_data = scaffold_split(tmp_data, seed=seed)

                    print(f"Start Training {model}")
                    train_and_evaluate(dataset_name, split_data, model, task, metric, seed=seed)
                print("\n")
            print("\n\n")
        except Exception as e:
            print(f"Error in {dataset_name}: {e}")
            continue

if __name__ == "__main__":
    coati_linear_probing_eval()
