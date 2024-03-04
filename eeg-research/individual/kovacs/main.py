"""
+----------------------------------------------------------------------------------------------------------------------+
AMBULATORY SEIZURE FORECASTING USING WEARABLE DEVICES
Main module (main.py)

Johannes Kepler University, Linz
ML Seminar/Practical Work 2023/24
Author:  Jozsef Kovacs
Created: 12/01/2024

The purpose of this project is to reproduce the deep learning approach to seizure forecasting from non-EEG signals,
collected from wrist-worn wearable devices that collect physiological patient parameters - as reported in the study
article:
    Nasseri, M., Pal Attia, T., Joseph, B., Gregg, N. M., Nurse, E. S., Viana, P. F., ... & Brinkmann, B. H. (2021).
    Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning.
    Scientific reports, 11(1), 21935.
    https://www.nature.com/articles/s41598-021-01449-2

The full reproduction of the experiments was not possible, as the dataset is only partially available (duration of
data collection, number of seizures). Furthermore, the above-mentioned paper did not have any acompanying codebase and
the authors did not disclose or provide code and specific details about the implementation of preprocessing, training,
testing (only the synthesis and general characterisation provided in the paper). Consequently, after analysis and a
number of preliminary explorative experiments with the obtained dataset, this project was built from scratch following
the descriptions and indications from the article. The dataset was obtained from: https://www.epilepsyecosystem.org/

Related Jupyter notebooks used as work drafts:
    explore_msg_data.ipynb  :   dataset content exploration and analysis
    prepoc_data.ipynb       :   dataset preparation and preprocessing methods

The solution consists of the following modules:
    main.py     : the executable Main module containing the preparing, training, evaluation and prediction algorithms
    datasets.py : contains the custom datasets, data loaders and auxiliary methods related to data loading
    trafo.py    : contains the transformations and data augmentation used in preprocessing
    model.py    : contains the neural network model definitions
    utils.py    : auxiliary and utility functions

The configuration is defined in the config.json file (to be customized as required).

Usage:  python main.py [--config <configuration_file>] [--mode <mode>] [--restore]

Command line arguments:
        <configuration_file>: the full path to the configuration file to use during execution (default is ./config.json)
        <mode>: should be one of the following:
            prepare  : creates a resized and augmented training dataset from the full sized original training images
            train    : trains the neural network (this is the default mode)
            finetune : finetunes the neural network
            eval     : evaluates the nerual network on a test set
            forecast : forecasting based on continuously fed data/timeframe by averaging predictions for windows
        --restore    : in 'train' mode this switch will restore the last saved model and continue training it
+----------------------------------------------------------------------------------------------------------------------+
"""

import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import datasets
from utils import console_msg
import importlib


def prepare(config: object):
    """
    This function pre-processes the data for use in training and evaluation. The data is provided for a given time
    period and with a given number of suitable inter-ictal and pre-ictal data segments. The suitable segments have
    to be preprocessed before directly used as input to the neural network. Preprocessing involves merging the data
    from different sensor files, calculating the additional FFT, SQI, hour-of-day channels and normalizing.

    :param config: the configuration object (specified in config.json)
    """
    metadata = dict()
    for folder, channels in config.wearable_data.sensor_channels.items():
        for ch in channels:
            metadata[(folder, ch)] = get_metadata(DATA_DIR, SUBJECT_ID, folder, ch)

    x = res[['BVP_BVP', 'TEMP_TEMP', 'EDA_EDA', 'HR_HR', 'ACC_Acc Mag']]
    tr = preproc_calculate_stft(x, freq=128, seqsize=4, overlap=3, step=5)

    # preprocess: calculate SQI for the EDA channel - rate of change in amplitude
    eda_sqi = eda_sqi_calc(np.asarray(res[['EDA_EDA']]).reshape(-1, ))

    # preprocess: calculate SQI for the BVP channel - spectral entropy
    bvp_sqi = bvp_sqi_spectral_entropy(np.asarray(res[['BVP_BVP']]).reshape(-1, ), window=4, segsize=60)

    # preprocess: calculate SQU for the ACC Mag channel - spectral power ratio
    accmag_sqi = accmag_sqi_spower_ratio(np.asarray(res[['ACC_Acc Mag']]).reshape(-1), freq=128, window=4, segsize=60,
                                         physio=(.8, 5.), broadband=(.8, 64.))

    # preprocess: add 24-hour part of the timestamp
    # [datetime.fromtimestamp(ts) for ts in np.asarray(res[['time']])]
    h24_info = np.asarray([datetime.fromtimestamp(ts / 1000.0).hour for ts in res[['time']].iloc[:, 0]])

    # concatenate input
    features = np.hstack([np.asarray(res)[:, 1:], np.repeat(tr, 128, axis=0)[:x.shape[0]], eda_sqi.reshape(-1, 1),
                          bvp_sqi.reshape(-1, 1), accmag_sqi.reshape(-1, 1), h24_info.reshape(-1, 1)])



# training method containing the training loop
def train(mode: str, config: object, model: object, train_loader: object, val_loader: object):
    """
    This function prepares and executes the neural network training loop. It implements two training
    phases (modes): 'train' and 'finetune'. These are done on separated training sets and their respective
    hyperparameters are specified separately in the config.json configuration file. This allows for using
    different batch size, learning rate, weight decay etc. for the training and finetuning phases.

    :param mode: 'train' for the training and 'finetune' for the fine-tuning phase
    :param config: the configuration object (specified in config.json)
    :param model: the wearable-LSTM model to be trained (if training is continued, then this is a restored model)
    :param train_loader: the dataloader used for loading the training dataset
    :param val_loader: the dataloader used for loading the validation dataset
    """
    # select the appropriate configuration block depending on the training mode
    tr_cfg = config.training_config if mode == 'train' else config.finetune_config

    # declare the TensorBoard summary writer to use during the training
    writer = SummaryWriter(log_dir=os.path.join(config.work_dir, 'tensorboard'))

    # the loss function used is the Binary Cross Entropy
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # the optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg.learning_rate, weight_decay=tr_cfg.weight_decay)

    # init TrainingState object
    tr_state = utils.TrainingState(model, config.model_file, tr_cfg)

    # init training loop variables for progress output
    num_epochs = tr_cfg.epochs + 1
    num_train_samples = train_loader.num_samples
    num_updates = int(num_train_samples / tr_cfg.batch_size)

    # the training loop
    console_msg("Starting training loop...\n")
    for tr_state.epoch in range(1, num_epochs):
        step_progess_bar = tqdm(total=num_updates, desc=tr_state.format_desc(), position=0)
        console_msg(f"STARTING EPOCH {tr_state.epoch}\n")
        model.train()
        tr_state.clear_train_losses()

        for batch in train_loader:
            optimizer.zero_grad()
            # execute training step
            features, targets, idxs = batch
            pred = model(features)
            # calculate loss
            tr_state.set_loss(loss_fn(pred, targets))
            # write scalars to board
            writer.add_scalar("training loss", tr_state.loss, tr_state.update_step)
            # increment the step variable
            tr_state.step()
            # update the progress bar
            step_progess_bar.set_description(desc=tr_state.format_desc(), refresh=True)
            step_progess_bar.update()
            # update the gradients
            tr_state.loss.backward()
            optimizer.step()

            # evaluate and log at specified periods
            if tr_state.update_step % tr_cfg.evaluate_at == 0:
                eval_and_save(tr_state, model, val_loader, writer, loss_fn)
                # early stopping
                if tr_state.stop_training:
                    console_msg(f'EARLY STOPPING - validation loss has not been improving'
                                f' for more than {tr_cfg.early_stop_threshold} update cycles.')
                    break

        # early stopping
        if tr_state.stop_training:
            break

        # evaluate and log at end of EPOCH
        eval_and_save(tr_state, model, val_loader, writer, loss_fn)
        console_msg(f'FINISHED EPOCH {tr_state.epoch}')


def eval_and_save(tr_state: utils.TrainingState, model, val_loader, writer, loss_fn):
    """
    This function is responsible for evaluation of the result and logging/writing out the metrics.

    :param tr_state: the TrainingState object holding the training state data and metrics
    :param model: the model being trained
    :param val_loader: the dataloader used for loading the validation/evaluation dataset
    :param writer: the TensorBoard writer to use for writing out the metrics
    :param loss_fn: the loss function to use
    """
    # evaluate the model and save it if it has improved
    tr_state.validation_loss = evaluate(model, val_loader, loss_fn)
    tr_state.save_if_best()
    tr_state.calulate_mean_train_loss()

    # write evaluation results to the console
    console_msg(tr_state.format_eval())

    # write metrics scalars to TensorBoard logs
    writer.add_scalars('loss', {'training': tr_state.mean_train_loss, 'validation': tr_state.validation_loss},
                       global_step=tr_state.update_step)

    # write model parameters to TensorBoard logs
    for i, param in enumerate(model.parameters()):
        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                             global_step=tr_state.update_step)

    # write model gradients to TensorBoard logs
    for i, param in enumerate(model.parameters()):
        if not param.grad is None:
            writer.add_histogram(tag=f'validation/gradients_{i}',
                                 values=param.grad.cpu(),
                                 global_step=tr_state.update_step)
    # flush written content
    writer.flush()


def evaluate(model, data_loader, loss_fn, config=None):
    """
    This function evaluates the model and optionally saves the predicted/target images and the pickle files
    for estimating server scores of the submission (submission/target pkl files for scoring_server.py)

    :param model: the CNN model to use for inference
    :param data_loader: the dataloader used for loading the evaluation/submission dataset
    :param loss_fn: the loss function to use for evaluation
    :param config: the configuration object (specified in config.json)
    """
    # set evaluation mode
    model.eval()
    # initialize validation losses
    val_losses = []

    # without gradient updates
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="...evaluate...", leave=False):
            # inference
            features, target, idxs = batch
            pred = model(features)
            # calculate and gather losses
            val_losses.append(loss_fn(pred, target))

        # calculate the mean validation loss
        val_loss = torch.stack(val_losses).mean().item()

    return val_loss


def test(config, model, test_loader):
    """
    This function executes the evaluation of the model on a separate test set.

    :param config: the configuration object (specified in config.json)
    :param model: the model being evaluated
    :param test_loader: the data loader used for the test (evaluation) dataset
    """
    console_msg("Starting testing loop...")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    test_loss = evaluate(model, test_loader, loss_fn, config=config)
    console_msg('Evaluation loss:', test_loss)


def forecast(config, model):
    """
    This function is used for forecasting from unseen data.

    :param config: the configuration object (specified in config.json)
    :param model: the model used for inference
    """
    # initialize the prediction data loader
    pred_dl = datasets.init_forecasting_loaders(config)
    # set evaluation mode
    model.eval()
    # without gradient updates
    with torch.no_grad():
        # the list to gather the result 1-D numpy arrays
        result_array_list = list()
        for batch in tqdm(pred_dl, desc="predict", position=0):
            # TODO: forecasting from coninuously read input by averaging over 4-min windows
            pass

    # write the pkl output
    utils.write_list_to_pkl(result_array_list, config.submission_pkl)



def main(args, config):
    """
    Main function which executes if the program is successfully started and the configuration is loaded.
    Depending on the mode of operation the main function will start the corresponding algorithms: preprocess,
    training, finetuning, evaluation or forecasting.

    :param args: command line arguments
    :param config: the configuation object (loaded from config.json)
    """
    # the mode of operation
    mode = args.mode
    # the model class to use (name defined in config.json) :: if there are model variations
    model_class_ = getattr(importlib.import_module("model"), config.network_config.model_class)

    if mode == 'preprocess':
        # the separate preparation phase
        console_msg('Preprocessing the input parquet files of subjects for training...')
        prepare(config)
    else:
        # initialize datasets and dataloaders
        # TODO: initialize the datasets and loaders
        train_dl, finetune_dl, val_dl, test_dl = datasets.init_datasets_and_loaders(mode, config)
        # model is restored - unless this is the initial training
        if args.restore or mode != 'train':
            # restore model state
            model = utils.retrieve_best_model(config, model_class_)
        else:
            # create new mode class - start from scratch
            model = model_class_(config.network_config)

        # move model to default device
        model.to(utils.get_default_device())

        if mode == 'train':
            # training mode
            console_msg('Training...')
            train(mode, config, model, train_dl, val_dl)
        elif mode == 'finetune':
            # finetuning mode
            console_msg('Finetuning...')
            train(mode, config, model, finetune_dl, val_dl)
        elif mode == 'eval':
            # evaluation mode
            console_msg('Evaluating...')
            test(config, model, test_dl)
        elif mode == 'forecast':
            # prediction mode
            console_msg('Forecasting - forecasting using unseen signals/interval...')
            forecast(config, model)


if __name__ == '__main__':
    from types import SimpleNamespace
    import argparse
    import json

    # set random seeds to constant value for reproducibility
    utils.set_random_seed()

    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Seizure forecasting from wearable devices', add_help=True)
    parser.add_argument('-c', '--config', type=str, default='./config.json',
                        help='the path and name of the config file')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='the mode of operation: prepare / train / eval / predict')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='restore last saved state and continue training (for mode: train)')
    args = parser.parse_args()

    console_msg('##################################################')
    console_msg('# SEIZURE FORECASTRING FROM WEARABLE DEVICES     #')
    console_msg('##################################################\n')

    try:
        # read the configuration into an object
        with open(args.config, 'r') as cfg_file_handle:
            config = json.load(cfg_file_handle, object_hook=lambda d: SimpleNamespace(**d))

    except FileNotFoundError:
        console_msg('ERROR: configuration file ' + args.config_file + ' does not exist!')
        console_msg('Please provide the configuration file or specify different path and filename')

    main(args, config)