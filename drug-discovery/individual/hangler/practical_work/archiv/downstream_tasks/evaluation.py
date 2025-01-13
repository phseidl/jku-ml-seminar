from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
# from ignite.engine import Events, Engine
# from ignite.metrics import Average, Loss
# from ignite.contrib.handlers import ProgressBar
# import gpytorch
# from gpytorch.mlls import VariationalELBO
# from gpytorch.likelihoods import GaussianLikelihood
# from due.dkl import DKL, GP, initial_values
# from due.fc_resnet import FCResNet

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from typing import Callable, Dict, Any
# from coati.models.regression.basic_due import basic_due

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
                res.append( function(y_true[s], y_pred[s], sample_weight=None if sample_weight is None else sample_weight[s]))#,
        return np.array(res)
    return wrapper

def davgp_score(y_true, y_pred, sample_weight=None):
    avgp = average_precision_score(y_true, y_pred, sample_weight=sample_weight)
    y_avg = np.average(y_true, weights=sample_weight)
    return avgp - y_avg

def perform_model_analysis(
        embeddings: np.ndarray, labels: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray,
        analysis_type: str, dataset_name: str, model_name: str, 
        assay_idx: int, model_details: Dict[str, Any]= None, 
        results_file: str = 'analysis_results.csv'
    ):
    """
    Perform model analysis using either logistic regression or DUE model.

    Args:
        embeddings (np.ndarray): The embeddings of the data.
        labels (np.ndarray): The labels corresponding to the embeddings.
        indices (np.ndarray): Indices used for additional reference or splits.
        train_idx (np.ndarray): Indices for training data.
        test_idx (np.ndarray): Indices for testing data.
        analysis_type (str): Type of analysis to perform ('logistic_regression' or 'due').
        dataset_name (str): Name of the dataset being analyzed.
        assay_idx (int): Assay number for the dataset.
        model_details (Dict[str, Any]): Details and callable functions for model analysis (example code in the beginning of this function).
        model_name (str): Name of the model used for embedding.
        results_file (str): Path to save the CSV results.
    """
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    if model_details is None:
        model_details = {}
        if analysis_type.lower() == 'logistic_regression' or analysis_type.lower() == 'logistic regression':
            model_details['params'] = {
                'max_iter': 1500,
                'class_weight': 'balanced',
                'C': 1,
                'random_state': 70135
            }
        elif analysis_type.lower() == 'due':
            model_details['params'] = {
                'x_train': X_train,
                'y_train': y_train,
                'x_test': X_test,
                'y_test': y_test,
                'continue_training': False,
                'steps': 1e4
            }
        else:
            raise ValueError("Unsupported analysis type")

    if analysis_type == 'logistic_regression':
        model = LogisticRegression(**model_details['params'])
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(embeddings[test_idx])[:, 1]
    elif analysis_type == 'due':
        model, model_results = due_model(**model_details['params'])
        model = model.to('cpu')
        y_pred, y_true, uncertainties = model_results
    else:
        raise ValueError("Unsupported analysis type")

    # Calculating scores
    AUROC_score = roc_auc_score(y_test, y_pred)
    roc_auc_std = bootstrap_metric(roc_auc_score, n=500)(y_test, y_pred).std()

    avgp = average_precision_score(y_test, y_pred)
    avgp_std = bootstrap_metric(average_precision_score, n=500)(y_test, y_pred).std()

    davgp = avgp - y_test.mean()
    davgp_std = bootstrap_metric(davgp_score, n=500)(y_test, y_pred).std()

    # Print the results to console
    print(f"dAP={davgp:.3f}, AUROC={AUROC_score:.3f}")

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Assay Index': [assay_idx],
        'Analysis Type': [analysis_type],
        'Model': [model_name],
        'AUROC Score': [AUROC_score],
        'AUROC std': [roc_auc_std],
        'avgp': [avgp],
        'avgp_std': [avgp_std],
        'davgp': [davgp],
        'davgp_std': [davgp_std]
    })

    # Save or append the results to a CSV file
    # Check if the file exists to decide whether to write headers or not
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, mode='w', header=True, index=False)


def due_model(
        x_train: np.ndarray, 
        y_train: np.ndarray,
        x_test: np.ndarray, 
        y_test: np.ndarray,
        save_as: str="due_model.pkl",
        load_as: str=None,
        continue_training: bool=False,
        batch_size: int=512,
        steps: int=1e4,
        depth: int=4,
        remove_spectral_norm: bool=False,
        random_seed: int=510,
    ):

    """
    Prepare data for DUE model and run the training process.
    
    Args:
        X_train (np.ndarray): Array of embeddings for training data.
        y_train (np.ndarray): Array of labels for training data.
        X_test (np.ndarray): Array of embeddings for testing data.
        y_test (np.ndarray): Array of labels for testing data.
        save_as (str): Path to save the model.
        load_as (str): Path to load the model.
        continue_training (bool): Whether to continue training from a loaded model.
        steps (int): Number of training steps.
        depth (int): Depth of the model.
        remove_spectral_norm (bool): Whether to remove spectral normalization from the model.
        random_seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed=random_seed)

    # data to torch.tensor
    train_x = torch.tensor(x_train, dtype=torch.float)
    train_y = torch.tensor(y_train, dtype=torch.float)
    test_x = torch.tensor(x_test, dtype=torch.float)
    test_y = torch.tensor(y_test, dtype=torch.float)

    # Dataset setup
    train_dataset = TensorDataset(train_x, train_y)    
    test_dataset = TensorDataset(test_x, test_y)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Code from here on from due.basic_due in COATI
    n_samples = train_x.shape[0]
    epochs = 1000

    input_dim = train_x.shape[-1]
    features = 256
    num_outputs = 1
    spectral_normalization = True
    coeff = 0.95
    n_inducing_points = 60
    n_power_iterations = 2
    dropout_rate = 0.03
    remove_spectral_norm=False,

    # ResFNN architecture
    feature_extractor = FCResNet(
        input_dim=input_dim,
        features=features,
        depth=depth,
        spectral_normalization=spectral_normalization,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        dropout_rate=dropout_rate,
    )
    kernel = "RBF"
    initial_inducing_points, initial_lengthscale = initial_values(
        train_dataset, feature_extractor, n_inducing_points
    )

    # Gaussian process (GP)
    gp = GP(
        num_outputs=num_outputs,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=kernel,
    )

    # Deep Kernel Learning (DKL) model
    model = DKL(feature_extractor, gp)

    likelihood = GaussianLikelihood()
    elbo_fn = VariationalELBO(likelihood, model.gp, num_data=len(train_dataset))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    lr = 1e-3
    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]
    parameters.append({"params": likelihood.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(parameters)

    def step(engine, batch):
        model.train()
        likelihood.train()
        optimizer.zero_grad()

        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        likelihood.eval()
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)
        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)

    metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())
    metric.attach(evaluator, "loss")

    if not load_as is None:
        read = torch.load(load_as)
        model.load_state_dict(read)

    if load_as is None or continue_training:
        print(f"Training with {n_samples} datapoints for {epochs} epochs")

        @trainer.on(Events.EPOCH_COMPLETED(every=int(epochs / 10) + 1))
        def log_results(trainer):
            evaluator.run(test_loader)
            print(
                f"Results - Epoch: {trainer.state.epoch} - "
                f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
                f"Loss: {trainer.state.metrics['loss']:.2f}"
            )

        trainer.run(train_loader, max_epochs=epochs)
        model.eval()
        likelihood.eval()
        torch.save(model.state_dict(), save_as)

    # If you want to differentiate the model.
    if remove_spectral_norm:
        model.feature_extractor.first = torch.nn.utils.remove_spectral_norm(
            model.feature_extractor.first
        )

    Xs_, Ys_, dYs_ = [], [], []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
        for batch_x, batch_y in test_loader:
            pred = model(batch_x.cuda())
            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()
            Xs_.append(batch_y.detach().cpu().numpy())
            Ys_.append(mean)
            dYs_.append(std)

    Xs = np.concatenate(Xs_, 0)
    Ys = np.concatenate(Ys_, 0)
    dYs = np.concatenate(dYs_, 0)

    return model, (Xs, Ys, dYs)
