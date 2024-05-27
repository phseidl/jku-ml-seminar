"""
+----------------------------------------------------------------------------------------------------------------------+
AMBULATORY SEIZURE FORECASTING USING WEARABLE DEVICES
Auxiliary and utility functions  (utils.py)

Johannes Kepler University, Linz
ML Seminar/Practical Work 2023/24
Author:  Jozsef Kovacs
Created: 17/02/2024

+----------------------------------------------------------------------------------------------------------------------+
"""

import random
from datetime import datetime
import numpy as np
import sklearn
import torch
import pickle as pkl
from pathlib import Path
import pandas as pd

from sklearn.metrics import roc_curve

# the random seed constant
RANDOM_SEED = 76


def ts_fmt():
    """
        This function returns an ISO formatted timestamp of the current datetime.

        :return: ISO formatted string containing the current timestamp
    """
    return datetime.now().isoformat(sep=' ', timespec='milliseconds')


def console_msg(m, *args):
    """
        Prints a message to the console preceded by the current timestamp.
    """
    print('[' + ts_fmt() + '] ' + str(m), *args)


def set_random_seed(rnd_seed=RANDOM_SEED):
    """
        Sets the randomizer seed to the specified constant value in the standard random function,
        the numpy random function and the PyTorch random function.

        :param rnd_seed: the value to set the random seed to (the default is the value of the RANDOM_SEED constant)
    """
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)


def get_default_device(config=None):
    """
        This auxiliary function returns the default device. If a configuration is provided and the device specified
        within is available, then it is returned as the default device. Without a specified configuration the function
        returns 'cuda' if available and 'cpu' otherwise.

        :param config: the configuration object specified in config.json containing the preferred 'device'
    """
    if config and config.device:
        if config.device.startswith('cuda') and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


def to_device(data, device):
    """
        Moves the data to the designated device.

        :param data: the data to be moved
        :param device: the device where the data should be moved
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def write_list_to_pkl(list_of_arrays, filepath):
    """
        Saves a list of arrays to a Pickle file.

        :param list_of_arrays: the list of arrays to save
        :param filepath: the path and filename
    """
    with open(filepath, 'wb') as f:
        pkl.dump(list_of_arrays, f)


def retrieve_best_model(config, ModelClass):
    """
        Retrieves the best model to the model class.

        :param config: the configuration object (from config.json)
        :param ModelClass: the model class to restore
    """
    model = ModelClass(config.network_config)
    model.load_state_dict(torch.load(config.model_file))
    model.to(get_default_device())
    return model


def retrieve_model(config, model_file, ModelClass):
    """
        Retrieves the best model to the model class.

        :param config: the configuration object (from config.json)
        :param file: the file containing the pytorch network model
        :param ModelClass: the model class to restore
    """
    model = ModelClass(config.network_config)
    model.load_state_dict(torch.load(model_file))
    #model.to(get_default_device())
    return model


def optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def metric_with_p_value(metric, y_true, y_pred_proba, n_permutations=10000, units_in_seg=60):
    actual_metric = metric(y_true, y_pred_proba)
    random_metric = []
    seg_labels = np.asarray(y_true)[::units_in_seg]
    count = 0
    for _ in range(n_permutations):
        permutes_segments = np.random.permutation(seg_labels)
        permuted_labels = permutes_segments.repeat(units_in_seg)
        permuted_metric = metric(permuted_labels, y_pred_proba)
        random_metric.append(permuted_metric)
        if permuted_metric >= actual_metric:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    random_mean = np.asarray(random_metric).mean()
    random_std = np.asarray(random_metric).std()
    return actual_metric, p_value, random_mean, random_std


class TrainingState(object):
    """
        This class holds the variables that describe the state of the training and contains methods, which facilitate
        updating and formatting, printing the current state of training to the console.
    """

    def __init__(self, model: object, model_dir: Path, model_file: str, train_cfg: object):
        super(TrainingState, self).__init__()

        # the model being trained
        self.model = model
        # the model file to store the best model so far
        self.model_file = model_file
        # epoch model dir
        self.model_dir = model_dir

        # the current epoch
        self.epoch = 0
        # the current update step
        self.update_step = 0
        # the current training loss
        self.loss = torch.tensor(.0)
        # the current validation loss
        self.validation_loss = float('inf')
        # the minimal validation loss so-far
        self.min_loss = float('inf')

        # early stopping parameters
        self.early_stopping = train_cfg.early_stopping
        self.early_stop_threshold = train_cfg.early_stop_threshold
        self.stop_training = False
        # non-improvement count
        self.non_improvement = 0

        # gathers the train losses
        self.train_losses = []
        self.mean_train_loss = float('inf')


    def format_desc(self):
        """
            Returns a formatted one-liner string of the provided training step parameters.
        """
        return f'epoch/step: [{self.epoch:02d}/{self.update_step:06d}] - tr.loss: {self.loss:7.5f},' \
               f' val.loss: {self.validation_loss:7.5f}, min.vl: {self.min_loss:7.5f}'

    def format_eval(self):
        """
            Returns a formatted one-liner string of the provided training step parameters at evaluation.
        """
        return f'epoch/step: [{self.epoch:02d}/{self.update_step:06d}] - mean training loss: {self.mean_train_loss:7.5f},' \
               f' val.loss: {self.validation_loss:7.5f}, min.vl: {self.min_loss:7.5f}'

    def save_end_of_epoch(self):
        device = next(self.model.parameters()).device
        self.model.to('cpu')
        torch.save(self.model.state_dict(), self.model_dir / Path(f"model_epoch_{self.epoch:05d}.net"))
        self.model.to(device)

    def save_if_best(self):
        """
            Calculates the minimum loss until now and saves the model state if the validation loss had improved.
        """
        if self.min_loss > self.validation_loss:
            self.min_loss = self.validation_loss
            self.non_improvement = 0
            torch.save(self.model.state_dict(), self.model_file)
        elif self.early_stopping:
            self.non_improvement += 1
            if self.non_improvement > self.early_stop_threshold:
                self.stop_training = True

    def step(self, increment=1):
        """
            Increments the training step counter.
        """
        self.update_step += increment

    def clear_train_losses(self):
        """
            Clears the train losses list.
        """
        self.train_losses = []

    def set_loss(self, new_loss):
        """
            Clears the train losses list.
        """
        self.loss = new_loss
        self.train_losses.append(self.loss.detach())

    def calulate_mean_train_loss(self):
        self.mean_train_loss = torch.stack(self.train_losses).mean().item()


class PredictionResult(object):
    """
        This class holds the results of prediction/forecasting.
    """

    def __init__(self, name: str):
        super(PredictionResult, self).__init__()
        self.name = name
        self.true_labels = []
        self.predicted_probs = []
        self.predicted_labels = []
        self.titles = []
        self.predictions = []
        self.result_summary = []

    def __len__(self):
        return len(self.true_labels)

    def register(self, true_label, pred_label, pred_prob, title, split_folder):
        self.true_labels.append(true_label)
        self.predicted_labels.append(pred_label)
        self.predicted_probs.append(pred_prob)
        self.titles.append(title)
        self.predictions.append([split_folder, true_label.item(), pred_label, pred_prob, title])

    def get_result_string(self, index: int = -1):
        return f" probability: {self.predicted_probs[index]:1.5f}, "\
               f"predicted label: {self.predicted_labels[index]}, true label: {self.true_labels[index]}, "\
               f"correct: {'[..OK..]' if self.predicted_labels[index] == self.true_labels[index] else '[......]'}"

    def calculate_adjusted_predictions(self):
        # determine optimal cutoff
        self.opt_cutoff = optimal_cutoff(self.true_labels, self.predicted_probs)[0]
        # re-evaluate with optimal cutoff
        self.adjusted_predictions = [1. if pp >= self.opt_cutoff else 0. for pp in self.predicted_probs]



    def report(self, result_title, nr_of_segments, split_folder, adjusted: bool = False):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(self.true_labels,
                                                          self.adjusted_predictions if adjusted else self.predicted_labels).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        total_cnt = len(self.predicted_labels)
        correct_cnt = tp + tn

        units_in_seg = len(self) // nr_of_segments
        n_permutations = 1000
        roc_auc, roc_pval, random_roc_mean, random_roc_std = metric_with_p_value(sklearn.metrics.roc_auc_score, self.true_labels, self.predicted_probs,
                                                n_permutations=n_permutations, units_in_seg=units_in_seg)
        ap_score, ap_pval, random_ap_mean, random_ap_std = metric_with_p_value(sklearn.metrics.average_precision_score, self.true_labels, self.predicted_probs,
                                                n_permutations=n_permutations, units_in_seg=units_in_seg)

        accuracy = (correct_cnt / total_cnt) * 100.0
        balanced_accuracy = ((specificity + sensitivity) / 2.) * 100.0

        self.result_summary.append([split_folder, tp, fp, fn, tn, sensitivity, specificity, total_cnt, correct_cnt, roc_auc,
                                    roc_pval, random_roc_mean, random_roc_std, ap_score, ap_pval, random_ap_mean,
                                    random_ap_std, self.opt_cutoff, accuracy, balanced_accuracy])

        s = ""
        s += "\n======================================================"
        s += f"\nRESULTS REPORT : {result_title}"
        s += "\n======================================================"
        s += f"\nROC AUC score: {roc_auc:3.2f} (p-value={roc_pval:1.5f} from {n_permutations} permutations)"
        s += f"\n   - random ROC AUC score: {random_roc_mean:3.2f} ({random_roc_std:3.2f})"
        s += f"\nAP score (precision-recall): {ap_score:3.2f} (p-value={ap_pval:1.5f} from {n_permutations} permutations)"
        s += f"\n   - random AP score: {random_ap_mean:3.2f} ({random_ap_std:3.2f})"
        s += f"\nOptimal cutoff threshold (ROC): {self.opt_cutoff:1.5f}"
        s += f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
        s += f"\nsensitivity: {sensitivity:1.5f}, specificity: {specificity:1.5f}"
        s += f"\ntotal_cnt: {total_cnt}, correct_cnt: {correct_cnt}"
        s += f"\naccuracy: {accuracy:3.2f}%"
        s += f"\nbalanced accuracy: {balanced_accuracy:3.2f}%"
        s += f"\nconfusion matrix:"
        s += f"\nP                   TRUE CLASS"
        s += f"\nR            | preictal | interictal| "
        s += f"\nE            +----------+-----------+"
        s += f"\nD   preictal |     {tp:>5d}|      {fp:>5d}|"
        s += f"\nI            |----------------------|"
        s += f"\nC interictal |     {fn:>5d}|      {tn:>5d}|"
        s += f"\nT            +----------------------+"
        s += f"\nE"
        s += f"\nD\n"
        s += "\nClassification report:"
        s += f"\n{sklearn.metrics.classification_report(self.true_labels, self.predicted_labels, target_names=['interictal', 'preictal'])}"
        s += "\n######################################################\n"

        return s
