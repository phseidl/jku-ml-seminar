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
import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from utils import console_msg, PredictionResult
import importlib
import preproc_utils as ppu
from pathlib import Path
from msg_subject_data import MsgDataHelper, MsgSubjectData
from datasets import DevDataLoader, MsgPredictionDataset, MsgTensorDataset
from torch.utils.data import DataLoader
from datetime import datetime
import click


def preprocess(config: object, mdh: MsgDataHelper):
    """
    This function pre-processes the data for use in training and evaluation. The data is provided for a given time
    period and with a given number of suitable inter-ictal and pre-ictal data segments. The suitable segments have
    to be preprocessed before directly used as input to the neural network. Preprocessing involves merging the data
    from different sensor files, calculating the additional FFT, SQI, hour-of-day channels and normalizing.

    :param config:  the configuration object (specified in config.json)
    :param mdh:     an auxiliary MsgDataHelper object used as the dataset descriptor and manager
    """
    # preprocess all subject data referenced in the MsgDataHelper
    for subject_id, subject_data in mdh.subject_data.items():
        console_msg(f"##############################################")
        console_msg(f"### start preprocessing subject ID: {subject_id}")

        # find suitable segments (lead seizure and interictal separation) and split 2:1 (train:test)
        subject_data.split_train_test(config.dataset_config.train_test_split, config.transforms_config.ratio)
        subject_data.print_preprocessing_summary()
        if not args.force_yes and not click.confirm('Do you want to continue with these settings?', default=True):
            exit()

        # data augmentation (preictal/training)
        if config.transforms_config.augment:
            subject_data.augmentation()

        # load previously saved scaler or calculate means and stds on the training data
        if subject_data.scaler is None or not config.dataset_config.reuse_scaler:
            console_msg("calculate mean and standard deviation for z-scoring...")
            subject_data.calculate_zscore_params(config)
        else:
            console_msg("using previously saved and now reloaded scaler for z-scoring...")

        # preprocess for each split
        for split in range(subject_data.num_split_combinations):

            # dataframe for preprocessing metadata (segment selection)
            pp_meta = pd.DataFrame(columns=['type', 'start', 'end', 'start_ts', 'end_ts', 'filename'])
            segment_index = 0

            # preprocess all data segments (train/test and preictal/interictal)
            console_msg(f">>> start preprocessing segments for: {subject_id}, split combination: {split}")

            # preprocess data segments
            for data_type in ['preictal_train', 'interictal_train', 'preictal_test', 'interictal_test']:
                file_cnt = 1
                console_msg(f"processing segment type: {data_type}, split combination: {split}")

                # preprocess the original segments - iterate over the suitable 1h intervals
                interval_list = subject_data.data[split][data_type] if data_type.startswith("preictal_") else subject_data.data[data_type]
                for i, interval in enumerate(interval_list):
                    start_ts, end_ts = interval
                    console_msg(f"processing '{data_type}' segment: {i}, start: {start_ts} - end: {end_ts}")
                    console_msg(
                        f" > from: {datetime.fromtimestamp(start_ts / 1000)} to: {datetime.fromtimestamp(end_ts / 1000)}")

                    # get the actual sensor data from the .parquet files
                    inp_data = subject_data.get_input_data(start_ts, end_ts)

                    if inp_data is None:
                        console_msg(f"ERROR: no data available for {subject_id} in this time interval...")
                        continue

                    # preprocess and save features to HDF file
                    preprocess_and_save_segment(split, subject_data, data_type, file_cnt, segment_index, start_ts, end_ts, inp_data, pp_meta)

                    segment_index += 1
                    file_cnt += 1

                # add the synthetic preictal segments - if data augmentation was applied
                if data_type == 'preictal_train' and not subject_data.augmented_preictal_train is None:
                    console_msg(f"processing synthetic preictal_train segments - DATA AUGMENTATION")
                    for key, inp_data in subject_data.augmented_preictal_train[split].items():
                        start_ts, end_ts, i = key
                        console_msg(f"processing augmented: {i}, start: {start_ts} - end: {end_ts}")
                        console_msg(
                            f" > from: {datetime.fromtimestamp(start_ts / 1000)} to: {datetime.fromtimestamp(end_ts / 1000)}")

                        # preprocess and save features to HDF file
                        preprocess_and_save_segment(split, subject_data, data_type, file_cnt, segment_index, start_ts, end_ts, inp_data, pp_meta)

                        segment_index += 1
                        file_cnt += 1

            # write preprocessing metadata to csv - will be used later by training/testing stages
            pp_meta.to_csv(mdh.preproc_dir / Path(subject_id) / Path(f'split_{split:03d}') / 'pp_metadata.csv', header=True)
            console_msg(f"--- finished preprocessing split {split} for subject: {subject_id}")

        console_msg(f"### finished preprocessing subject: {subject_id}")
        console_msg(f"##############################################\n")


def preprocess_and_save_segment(split: int, subject_data: MsgSubjectData, data_type: str, file_cnt: int,
                                segment_index: int, start_ts, end_ts, data: pd.DataFrame, pp_meta: pd.DataFrame):
    """
    Auxiliary function for calling preprocessing on a given 1-hour data segment, standardizing the obtained features
    and writing the preprocessed output to HDF files.

    :param split:        the index of the preictal split combination belonging to this preprocessing run
    :param subject_data: the auxiliary subject data descriptor object
    :param data_type:    the type of the data segment (e.g. 'preictal_test')
    :param cnt:          the index of the current segment (and file) in the sequence of processed segments
    :param data:         the raw data features of the segment (time (1) + signals (8) channels)
    """

    # preprocess (FFT, SQI, added 24h time - extends data with additional features)
    console_msg(" > start preprocessing to obtain additional features...")
    features = ppu.preprocess(config, data)
    console_msg(" > finished preprocessing...")

    # standardize and convert to dataframe
    console_msg(" > z-scoring / standardize...")
    df = pd.DataFrame(subject_data.scaler[split].transform(features))

    # write preprocessed result to HDF
    path = subject_data.mdh.preproc_dir / Path(subject_data.subject_id) / Path(f'split_{split:03d}') / Path(data_type)
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f'{data_type}_{file_cnt:03d}.h5'
    console_msg(f" > start writing prepocessed data to: {filename}")
    df.to_hdf(filename, key='df', mode='w', complevel=9, complib='bzip2', format='table')
    console_msg(f" > finish writing prepocessed data to: {filename}")

    # update preprocessed metadata table
    pp_meta.loc[segment_index] = [data_type, datetime.fromtimestamp(start_ts / 1000),
                                  datetime.fromtimestamp(end_ts / 1000), start_ts, end_ts, filename]


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

    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=tr_cfg.scheduler_step, gamma=tr_cfg.scheduler_gamma)

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
            if tr_cfg.evaluate_at != 0 and tr_state.update_step % tr_cfg.evaluate_at == 0:
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

        # adjust learning rate
        scheduler.step()


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


def forecast(config, args, model_class_, mdh, split_folder):
    """
    This function is used for forecasting from unseen data.

    :param config: the configuration object (specified in config.json)
    :param model: the model used for inference
    """
    subject_id = args.subjects[0]
    console_msg("#" + "-" * 120)
    console_msg(f"# FORECASTING     subject: {subject_id}")
    console_msg("#" + "-" * 120)

    # split and prepare training and validation loaders
    train_work_dir = Path(config.project_root) / Path(config.train_dir) / Path(subject_id) / Path(split_folder)
    model_file = Path(train_work_dir) / Path(config.model_dir) / Path(args.model_file)
    console_msg(f" > retrieve model to use for forecasting: {model_file} ")
    model = utils.retrieve_model(config, model_file, model_class_)

    # forecasting dataset
    forecast_ds = MsgPredictionDataset(mdh, args.subjects[0], split_folder)
    console_msg(' > Forecasting - forecasting using unseen segments/intervals...')
    console_msg(f" > dataset lenght: {len(forecast_ds)} x {args.forecast_segsize}-min recordings")

    # set evaluation mode
    model.eval()
    model.to(config.device)
    model.float()
    # without gradient updates
    with torch.no_grad():
        # the list to gather the result 1-D numpy arrays
        results = {'unit': PredictionResult('unit'), 'window': PredictionResult('window'), 'segment': PredictionResult('segment')}
        window = config.eval_window
        threshold = config.eval_threshold
        console_msg(f" - segment length:      {args.forecast_segsize:>5d}")
        console_msg(f" - window length:       {window:>5d}")
        console_msg(f" - default threshold:   {threshold:>8.2f}")

        for segment, label, data_type, filename, start, end in tqdm(forecast_ds):
            nr_units_in_segment = segment.shape[0]
            hidden = None
            result_array_list = list()
            window_count = segment.shape[0] // window
            start_window = int(window_count * ((nr_units_in_segment - args.forecast_segsize) / nr_units_in_segment))
            for part_idx in range(start_window, window_count):

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
                    pred_label = round(forecast_ds.LABEL_1 if act_pred.item() > threshold else forecast_ds.LABEL_0)
                    positive += pred_label
                    # register result for the given unit
                    results['unit'].register(torch.round(label), pred_label, act_pred.item(),
                                             f"{idx + 1:02d} at T-{nr_units_in_segment-idx-1 + mdh.preictal_setback:02d}min",
                                             split_folder)

                # register forecasting results for the window (mean over units)
                mean_prob /= window * 1.0
                results['window'].register(torch.round(label), round((forecast_ds.LABEL_1 if mean_prob >= threshold else forecast_ds.LABEL_0)),
                                           mean_prob, f"{part_idx + 1:02d} of length {window}m", split_folder)
                result_array_list.append(mean_prob)

            # register forecasting results for the segment (max over window means)
            forecast_prob = max(result_array_list)
            results['segment'].register(torch.round(label), round((forecast_ds.LABEL_1 if forecast_prob > threshold else forecast_ds.LABEL_0)),
                                        forecast_prob, f" from: {start} to: {end}, data type: {data_type}, filename: {Path(filename).name}", split_folder)

        # adjust predicted labels with opt.cutoff from ROC
        for res in results.values():
            res.calculate_adjusted_predictions()

        # detailed report of predictions
        print("\n#" + "-" * 120)
        print("#   D E T A I L E D   R E P O R T                    #")
        print("#" + "-" * 120 + "\n")

        for seg_idx, seg_title in enumerate(results['segment'].titles):
            print("#" * 120)
            print(f"<<< START SEGMENT {seg_title}")
            window_count = args.forecast_segsize // window
            start_window = int(window_count * ((60 - args.forecast_segsize) / 60))
            for part_idx in range(start_window, window_count):
                print(f"\n<<< START WINDOW {results['window'].titles[seg_idx * window_count + part_idx]} >>>")
                for seq_idx in range(window):
                    idx = nr_units_in_segment * seg_idx + window * part_idx + seq_idx
                    print(f"...unit {results['unit'].titles[idx]} | {results['unit'].get_result_string(idx)}")
                print(f"<<< END WINDOW {results['window'].titles[seg_idx * window_count + part_idx]} | {results['window'].get_result_string(seg_idx * window_count + part_idx)} >>>")

            print( "#" + "-" * 120)
            print(f"# <<< END SEGMENT | {results['segment'].get_result_string(seg_idx)} >>>")
            print( "#" + "-" * 120 + "\n\n")

        # print results report and write result and summary files
        for key, res in results.items():
            print("\n" + res.report(key, len(results['segment']), split_folder, adjusted=True) + "\n")

        return results


def write_forecasting_results(args, config, res_dict):

    results_dir = Path(config.project_root) / Path(config.results_dir) / Path(args.subjects[0])
    Path.mkdir(results_dir, parents=True, exist_ok=True)

    summaries_dict = {'unit': [], 'window': [], 'segment': []}
    predictions_dict = {'unit': [], 'window': [], 'segment': []}
    for key in ['unit', 'window', 'segment']:
        for split_folder, results in res_dict.items():
            summaries_dict[key] += results[key].result_summary
            predictions_dict[key] += results[key].predictions

        pred_df = pd.DataFrame(predictions_dict[key],
                               columns=['split', 'true_label', 'predicted_label', 'predicted_prob', 'remark'])
        summary_df = pd.DataFrame(summaries_dict[key],
                                  columns=['split', 'TP', 'FP', 'FN', 'TN', 'sensitivity', 'specificty',
                                           'total', 'correct', 'ROC AUC', 'ROC AUC p-value',
                                           'random ROC AUC (mean)', 'random ROC AUC (std)',
                                           'AP', 'AP p-value', 'random AP (mean)', 'random AP (std)',
                                           'opt.cutoff', 'accuracy', 'balanced accuracy'])
        pred_df.to_csv(Path(results_dir / Path(f"{key}_predictions.csv")))
        summary_df.to_csv(Path(results_dir / Path(f"{key}_summary.csv")))


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
        if config.cross_validation:
            path = Path(config.project_root) / Path(config.preproc_dir) / Path(args.subjects[0])
            split_combinations = [sf.name for sf in list(path.glob("split_*"))]
        else:
            split_combinations = ['split_000']

        for split_folder in split_combinations:
            # restore saved model and continue training
            if args.restore:
                train_work_dir = Path(config.project_root) / Path(config.train_dir) / Path(args.subjects[0]) / Path(split_folder)
                model_file = Path(train_work_dir) / Path(config.model_dir) / Path(args.model_file)
                model = utils.retrieve_model(config, model_file, model_class_)
            else:
                # create new mode class - start from scratch
                model = model_class_(config.network_config)

            # move model to default device
            model.to(utils.get_default_device())

            # training mode
            console_msg('Training...')
            # initialize and load the "in-memory" dataset
            dataset = MsgTensorDataset(mdh, args.subjects[0], split_folder, sequence_length=60)

            # split and prepare training and validation loaders
            val_size = int(len(dataset) * config.dataset_config.valid_ratio)
            train_size = len(dataset) - val_size
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
            model = model.float()
            model.to(config.device)
            train_dl = DevDataLoader(DataLoader(train_ds, batch_size=config.training_config.batch_size, shuffle=True), train_size, torch.device(config.device))
            valid_dl = DevDataLoader(DataLoader(val_ds, batch_size=config.training_config.batch_size, shuffle=False), val_size, torch.device(config.device))

            # prepare training work directory
            train_work_dir = Path(config.project_root) / Path(config.train_dir) / Path(args.subjects[0]) / Path(split_folder)
            train_work_dir.mkdir(parents=True, exist_ok=True)

            # execute the training loop
            train(mode, config, model, train_dl, valid_dl, train_work_dir)

    elif mode == 'finetune':
        # finetuning mode
        console_msg('Finetuning...')
        #train(mode, config, model, finetune_dl, val_dl)

    elif mode == 'forecast':
        if config.cross_validation:
            path = Path(config.project_root) / Path(config.preproc_dir) / Path(args.subjects[0])
            split_combinations = [sf.name for sf in list(path.glob("split_*"))]
        else:
            split_combinations = ['split_000']

        res_dict = {}
        for split_folder in split_combinations:
            res_dict[split_folder] = forecast(config, args, model_class_, mdh, split_folder)

        write_forecasting_results(args, config, res_dict)

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
    parser.add_argument('-g', '--forecast_segsize', type=int, default=60,
                        help='the size of the segment considered at forecasting')
    parser.add_argument('-y', '--force_yes', action='store_true',
                        help='continue assuming yes was answered to prompts')
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
