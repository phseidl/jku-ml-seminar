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
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from utils import console_msg, PredictionResult
import importlib
import preproc_utils as ppu
from pathlib import Path
from msg_subject_data import MsgDataHelper
from datasets import MsgTrainDataset, MsgDatasetInMem, DevDataLoader, MsgPredictionDataset
from torch.utils.data import DataLoader
from datetime import datetime


def preprocess(config: object, mdh: MsgDataHelper):
    """
    This function pre-processes the data for use in training and evaluation. The data is provided for a given time
    period and with a given number of suitable inter-ictal and pre-ictal data segments. The suitable segments have
    to be preprocessed before directly used as input to the neural network. Preprocessing involves merging the data
    from different sensor files, calculating the additional FFT, SQI, hour-of-day channels and normalizing.

    :param config: the configuration object (specified in config.json)
    """
    # preprocess all subject data referenced in the MsgDataHelper
    for subject_id, subject_data in mdh.subject_data.items():
        console_msg(f"##############################################")
        console_msg(f"### start preprocessing subject ID: {subject_id}")

        # find suitable segments and split 2:1
        subject_data.split_train_test(2/3., 6.)

        # calculate means and stds on the training data
        console_msg("calculate mean and standard deviation for z-scoring...")
        subject_data.calculate_zscore_params()

        # dataframe for preprocessing metadata (segment selection)
        pp_meta = pd.DataFrame(columns=['type', 'start', 'end', 'start_ts', 'end_ts', 'filename'])
        file_cnt = 0

        # preprocess all data segments (train/test and preictal/interictal)
        console_msg(f">>> start preprocessing segments for: {subject_id}")
        for data_type, intervals in subject_data.data.items():
            console_msg(f"processing segment type: {data_type}")
            # iterate over the suitable 1h intervals
            cnt = 0
            for i, interval in enumerate(intervals):
                start_ts, end_ts = interval
                console_msg(f"processing '{data_type}' segment: {i}, start: {start_ts} - end: {end_ts}")
                console_msg(f" > from: {datetime.fromtimestamp(start_ts/1000)} to: {datetime.fromtimestamp(end_ts/1000)}")

                # get the actual sensor data from the .parquet files
                inp_data = subject_data.get_input_data(start_ts, end_ts)

                if inp_data is None:
                    console_msg(f"ERROR: no data available for {subject_id} in this time interval...")
                    continue

                for rep in range(config.transforms_config.ratio if data_type == 'preictal_train' else 1):
                    # for rep in range(1):
                    aug = "(augmented version " + str(rep) + ")" if rep > 0 else "(original version)"
                    console_msg(f"{aug}")

                    data = inp_data
                    if rep > 0:
                        data = ppu.add_noise3(data, config.wearable_data.freq)

                    # preprocess (FFT, SQI, added 24h time)
                    console_msg(" > start preprocessing to obtain additional features...")
                    features = ppu.preprocess(config, data)
                    features = np.nan_to_num(features)
                    console_msg(" > finished preprocessing...")

                    # normalize and convert to dataframe
                    console_msg(" > z-scoring...")
                    df = pd.DataFrame(ppu.normalize(features, subject_data.means, subject_data.stds))

                    # write result to HDF
                    path = mdh.preproc_dir / Path(subject_id) / Path(data_type)
                    path.mkdir(parents=True, exist_ok=True)
                    filename = path / f'{data_type}_{cnt+1:03d}.h5'
                    console_msg(f" > start writing prepocessed data to: {filename}")
                    df.to_hdf(filename, key='df', mode='w', complevel=9, complib='bzip2', format='table')
                    console_msg(f" > finish writing prepocessed data to: {filename}")

                    pp_meta.loc[file_cnt] = [data_type, datetime.fromtimestamp(start_ts/1000), datetime.fromtimestamp(end_ts/1000), start_ts, end_ts, filename]
                    file_cnt += 1
                    cnt += 1

        # write preprocessing metadata to csv
        pp_meta.to_csv(mdh.preproc_dir / Path(subject_id) / 'pp_metadata.csv', header=True)
        console_msg(f"### finished preprocessing subject: {subject_id}")
        console_msg(f"##############################################\n")


# training method containing the training loop
def train(mode: str, config: object, model: object, train_loader: object, val_loader: object, train_dir: Path):
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
    tensorboard_dir = train_dir / Path('tensorboard')
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # the loss function used is the Binary Cross Entropy
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # the optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg.learning_rate, weight_decay=tr_cfg.weight_decay)

    # init TrainingState object
    model_dir = train_dir / Path(config.model_dir)
    model_dir.mkdir(exist_ok=True)
    tr_state = utils.TrainingState(model, model_dir, model_dir / Path(config.model_file), tr_cfg)

    # init training loop variables for progress output
    num_epochs = tr_cfg.epochs + 1
    num_train_samples = len(train_loader)
    num_updates = int(num_train_samples)

    # the training loop
    console_msg("Starting training loop...\n")
    for tr_state.epoch in range(1, num_epochs):
        step_progess_bar = tqdm(total=num_updates, desc=tr_state.format_desc(), position=0)
        console_msg(f"STARTING EPOCH {tr_state.epoch}\n")
        tr_state.clear_train_losses()

        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            # execute training step
            features, targets = batch
            pred, _ = model(features.float())

            # calculate loss
            targets = torch.unsqueeze(targets, 1)
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
        eval_and_save(tr_state, model, val_loader, writer, loss_fn, end_of_epoch=True)
        console_msg(f'FINISHED EPOCH {tr_state.epoch}')


def eval_and_save(tr_state: utils.TrainingState, model, val_loader, writer, loss_fn, end_of_epoch=False):
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
    if end_of_epoch:
        tr_state.save_end_of_epoch()
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
            features, target = batch
            pred, _ = model(features.float())
            # calculate and gather losses
            val_losses.append(loss_fn(pred, torch.unsqueeze(target, 1)).detach())

        # calculate the mean validation loss
        val_loss = torch.stack(val_losses).mean().item()

    return val_loss


def forecast(config, model, forecast_ds):
    """
    This function is used for forecasting from unseen data.

    :param config: the configuration object (specified in config.json)
    :param model: the model used for inference
    """
    # set evaluation mode
    model.eval()
    model.to(config.device)
    model.float()
    # without gradient updates
    with torch.no_grad():
        # the list to gather the result 1-D numpy arrays
        results = {'unit': PredictionResult(), 'window': PredictionResult(), 'segment': PredictionResult()}
        window = config.eval_window
        threshold = config.eval_threshold
        console_msg(f" - window length: {window:>5d}")
        console_msg(f" - threshold val: {threshold:>3.2f}")
        for segment, label, data_type, filename, start, end in forecast_ds:
            console_msg(f" >>> data type: {data_type}, filename: {filename}")
            console_msg(f" >>> from: {start} to: {end}")
            hidden = None
            result_array_list = list()
            for part_idx in range(segment.shape[0] // window):
                mean_prob, max_prob, positive, total = 0., 0., 0, 0
                for seq_idx in range(window):
                    total += 1
                    idx = window * part_idx + seq_idx
                    input = segment[idx:idx+1, :, :]
                    input = utils.to_device(input, torch.device(config.device))
                    pred, (h_0, c_0) = model(input.float(), hidden)
                    hidden = (h_0.detach(), c_0.detach())
                    act_pred = torch.special.expit(pred).cpu().detach()
                    mean_prob += act_pred.item()
                    max_prob = max(max_prob, act_pred)
                    console_msg(f"{idx + 1:02d}:: prob: {act_pred.item():1.5f} vs. label: {torch.round(label)}")
                    pred_label = round(forecast_ds.LABEL_1 if act_pred.item() > threshold else forecast_ds.LABEL_0)
                    positive += pred_label
                    results['unit'].register(torch.round(label), pred_label, act_pred.item())

                mean_prob /= window * 1.0
                results['window'].register(torch.round(label), round((forecast_ds.LABEL_1 if max_prob >= threshold else forecast_ds.LABEL_0)), mean_prob)
                result_array_list.append(mean_prob)
                console_msg(f"{part_idx+1:02d}. {window}m-seg: mean prob:", mean_prob)

            forecast_prob = max(result_array_list)
            results['segment'].register(torch.round(label), round((forecast_ds.LABEL_1 if forecast_prob > threshold else forecast_ds.LABEL_0)), forecast_prob)
            console_msg("\n====\n", f" <<FORECAST>> max prob: {forecast_prob:1.5f} vs. label: {torch.round(label)}", "\n====\n")

        for key, res in results.items():
            print("\n" + res.report(key) + "\n")



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

    # initialize the MsgDataHelper for this config
    mdh = MsgDataHelper(config, args.subjects)

    if mode == 'preprocess':
        # the separate preparation phase
        console_msg('Preprocessing the input parquet files of subjects for training...')
        preprocess(config, mdh)

    elif mode == 'train':
        # create new mode class - start from scratch
        model = model_class_(config.network_config)

        # move model to default device
        model.to(utils.get_default_device())

        # training mode
        console_msg('Training...')
        # initialize and load the "in-memory" dataset
        dataset = MsgDatasetInMem(mdh, args.subjects[0])

        # split and prepare training and validation loaders
        val_size = int(len(dataset) * config.dataset_config.valid_ratio)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        model = model.float()
        model.to(config.device)
        train_dl = DevDataLoader(DataLoader(train_ds, batch_size=config.training_config.batch_size, shuffle=True), train_size, torch.device(config.device))
        valid_dl = DevDataLoader(DataLoader(val_ds, batch_size=config.training_config.batch_size, shuffle=False), val_size, torch.device(config.device))

        # prepare training work directory
        train_work_dir = Path(config.project_root) / Path(config.train_dir) / Path(args.subjects[0])
        train_work_dir.mkdir(exist_ok=True)

        # execute the training loop
        train(mode, config, model, train_dl, valid_dl, train_work_dir)

    elif mode == 'finetune':
        # finetuning mode
        console_msg('Finetuning...')
        #train(mode, config, model, finetune_dl, val_dl)

    elif mode == 'forecast':

        console_msg('Training...')
        # split and prepare training and validation loaders
        train_work_dir = Path(config.project_root) / Path(config.train_dir) / Path(args.subjects[0])
        model_file = Path(train_work_dir) / Path(config.model_dir) / Path(args.model_file)
        model = utils.retrieve_model(config, model_file, model_class_)

        console_msg("-------------------")
        console_msg("# FORECASTING     #")
        console_msg("-------------------")
        # forecasting dataset
        forecast_ds = MsgPredictionDataset(mdh, args.subjects[0])
        console_msg('Forecasting - forecasting using unseen segments/intervals...')
        console_msg(f"dataset lenght: {len(forecast_ds)} x 1-hour recordings")

        forecast(config, model, forecast_ds)

    else:
        console_msg(f'ERROR: mode not implemented: {mode}')


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
                        help='the mode of operation: preprocess / train / forecast')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='restore last saved state and continue training (for mode: train)')
    parser.add_argument('-s', '--subjects', nargs='*', type=str, default=None,
                        help='subject identifier(s) to process (eg. MSEL_01676)')
    parser.add_argument('-f', '--model_file', type=str, default='best_model.net',
                        help='the saved model file to use in evaluation and forecasting')
    args = parser.parse_args()

    console_msg('##################################################')
    console_msg('# SEIZURE FORECASTING USING WEARABLE DEVICES     #')
    console_msg('##################################################\n')

    try:
        # read the configuration into an object
        with open(args.config, 'r') as cfg_file_handle:
            config = json.load(cfg_file_handle, object_hook=lambda d: SimpleNamespace(**d))

    except FileNotFoundError:
        console_msg('ERROR: configuration file ' + args.config_file + ' does not exist!')
        console_msg('Please provide the configuration file or specify different path and filename')

    main(args, config)
