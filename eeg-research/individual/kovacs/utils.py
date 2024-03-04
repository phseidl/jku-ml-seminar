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
import torch
import pickle as pkl

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


class TrainingState(object):
    """
        This class holds the variables that describe the state of the training and contains methods, which facilitate
        updating and formatting, printing the current state of training to the console.
    """

    def __init__(self, model: object, model_file: str, train_cfg: object):
        super(TrainingState, self).__init__()

        # the model being trained
        self.model = model
        # the model file to store the best model so far
        self.model_file = model_file

        # the current epoch
        self.epoch = 0
        # the current update step
        self.update_step = 0
        # the current training loss
        self.loss = torch.tensor(.0)
        # the current validation loss
        self.validation_loss = np.float('inf')
        # the minimal validation loss so-far
        self.min_loss = np.float('inf')

        # early stopping parameters
        self.early_stopping = train_cfg.early_stopping
        self.early_stop_threshold = train_cfg.early_stop_threshold
        self.stop_training = False
        # non-improvement count
        self.non_improvement = 0

        # gathers the train losses
        self.train_losses = []
        self.mean_train_loss = np.float('inf')


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
            Increments the training
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
        self.train_losses.append(self.loss)

    def calulate_mean_train_loss(self):
        self.mean_train_loss = torch.stack(self.train_losses).mean().item()