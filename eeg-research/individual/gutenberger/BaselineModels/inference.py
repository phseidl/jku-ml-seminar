# check successful setup
import torch
import os
import yaml
import math
import csv
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from torch import from_numpy as np2TT

from BaselineModels.utils.misc import model_select
from BaselineModels.utils.build import create_dataset
from UPT4EEG.evaluation.eval_metrics import calc_eval_metrics

if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.cuda.get_device_name(device))

print(os.environ.get("CUDA_VISIBLE_DEVICES"))

# Create the parser
parser = argparse.ArgumentParser(description="Inference script to generate eval metrics for the baseline models.")

# Define the argument
parser.add_argument('model_class', type=str, help="Model class the eval metrics should be calculated for.")
parser.add_argument('use_montage', type=str, help="Defines which montage to use in inference ('tuh', 'tuh_rand' or 'random').")

# Parse the arguments
args = parser.parse_args()

MODEL_CLASS = args.model_class
use_montage = args.use_montage

DATASET = 'TUH'
workin_dir = '/system/user/studentwork/gutenber'

#MODEL_CLASS = 'CLEEGN'
#MODEL_FILE_NAME = 'CLEEGN_Jan09_07-08-04.pth'
#MODEL_CLASS = 'OneD_Res_CNN'
#MODEL_FILE_NAME = 'OneD_Res_CNN_Jan08_15-39-06.pth'

if MODEL_CLASS == 'IC_U_Net':
    #MODEL_FILE_NAME = 'IC_U_Net_Jan08_14-43-36.pth'
    MODEL_FILE_NAME = 'IC_U_Net_Jan22_19-36-59.pth'    #new sweep, 4s
elif MODEL_CLASS == 'OneD_Res_CNN':
    #MODEL_FILE_NAME = 'OneD_Res_CNN_Jan08_15-39-06.pth'
    MODEL_FILE_NAME = 'OneD_Res_CNN_Jan21_14-45-51.pth' 
elif MODEL_CLASS == 'CLEEGN':
    #MODEL_FILE_NAME = 'CLEEGN_Jan21_12-08-17.pth'    #trained on 1s
    MODEL_FILE_NAME = 'CLEEGN_Jan09_07-08-04.pth'   #trained on 4s
    MODEL_FILE_NAME = 'CLEEGN_Jan22_17-49-22.pth' #trained on 4s, correct frequency in model dimension

#use_montage = 'tuh'
downsample = False

if MODEL_CLASS == 'OneD_Res_CNN':
    window_size = 1 #1.6 #cfg_dataset["window_size"]
    if downsample:
        window_size = 2
elif MODEL_CLASS == 'IC_U_Net':
    window_size = 4
    if downsample:
        window_size = 8
elif MODEL_CLASS == 'CLEEGN':
    window_size = 1
else:
    window_size = 4
    print('TODO: Unknown model class.')
stride = window_size/2



csv_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/eval_metrics_'+ MODEL_CLASS + '_' + use_montage + '_downsample_' + str(downsample) + '.csv'
csv_summary_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/subject_metrics_summary_'+ MODEL_CLASS + '_' + use_montage  + '_downsample_' + str(downsample) +'.csv'
final_summary_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/metrics_summary_'+ MODEL_CLASS + '_' + use_montage  + '_downsample_' + str(downsample) + '_' + str(window_size) + 's.csv'
model_path = os.path.join(workin_dir, 'logs/TUH', MODEL_CLASS, MODEL_FILE_NAME)
config_path = os.path.join(workin_dir, 'configs/config_baseline_models.yml')
model_config_path = os.path.join(workin_dir, 'configs/model_config.yml')
#config_path = '/system/user/studentwork/gutenber/configs/config.yml'
#model_config_path = '/system/user/studentwork/gutenber/configs/model_config.yml'

state_path = os.path.join(model_path)
state = torch.load(state_path, map_location="cpu")

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())
cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]

SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
#use_montage = cfg_dataset['use_montage']
#use_montage = 'user_specific'

#NUM_EPOCHS = 1 #cfg_general['epochs']
#BATCH_SIZE = cfg_model['batch_size']
#LR         = cfg_model["learning_rate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_met_per_sub = pd.DataFrame()
mse_list_total = []
r2_list_total = []
snr_db_list_total = []
pcc_list_total = []

for subject in cfg_dataset["subjects_test"]:
    x_test, y_test = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        [subject], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride,
        use_montage=use_montage, downsample=downsample
    )

    x_test = np2TT(np.expand_dims(x_test, axis=1))
    y_test = np2TT(np.expand_dims(y_test, axis=1))

    testset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=True)

    model = model_select(MODEL_CLASS, cfg_model, cfg_dataset, window_size, downsample=downsample).to(device)

    model.load_state_dict(state["state_dict"])

    ################################## EVALUATION METRICS ##########################

    

    total_updates = len(test_loader)

    test_loss_total = 0
    r2_metric_total = 0
    snr_total = 0
    snr_db_total = 0
    invalid_cnt = 0
    pcc_total = 0
    cc_total = 0
    rmse_total = 0
    rrmse_total = 0
    r2_list = []
    mse_list = []
    snr_list = []
    snr_db_list = []
    pcc_list = []
    cc_list = []
    rrmse_list = []
    rmse_list = []

    pbar = tqdm(total=total_updates)
    pbar.update(0)

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)
        with torch.no_grad():
            y_hat = model(x_batch)
        eval_metrics = calc_eval_metrics(y_hat, y_batch)
        test_loss_total += eval_metrics['MSE']
        r2_metric_total += eval_metrics['R2']
        pcc_total += eval_metrics['PCC']
        cc_total += eval_metrics['CC']
        rrmse_total += eval_metrics['RRMSE']
        rmse_total += eval_metrics['RMSE']
        snr_total += eval_metrics['SNR']
        snr_db_total += eval_metrics['SNR_dB']
        mse_list.append(eval_metrics['MSE'])
        r2_list.append(eval_metrics['R2'])
        snr_list.append(eval_metrics['SNR'])
        snr_db_list.append(eval_metrics['SNR_dB'])
        pcc_list.append(eval_metrics['PCC'])
        cc_list.append(eval_metrics['CC'])
        rrmse_list.append(eval_metrics['RRMSE'])
        rmse_list.append(eval_metrics['RMSE'])

        pbar.update()
        pbar.set_description(
            f"MSE loss: {eval_metrics['MSE']:.4f}, "
            f"SNR: {eval_metrics['SNR']:.4f}, "
            f"SNR in dB: {eval_metrics['SNR_dB']:.4f}, "
            f"R^2 score: {eval_metrics['R2']:.4f}, "
            f"PCC: {eval_metrics['PCC']:.4f}, "
            f"CC: {eval_metrics['CC']:.4f}, "
            f"RRMSE: {eval_metrics['RRMSE']:.4f}, "
            f"RMSE: {eval_metrics['RMSE']:.4f}"
        )
        
    test_loss_total /= len(test_loader)
    r2_metric_total /= len(test_loader)
    snr_total /= (len(test_loader))
    snr_db_total /= (len(test_loader))
    rrmse_total /= len(test_loader)
    rmse_total /= len(test_loader)
    pcc_total /= len(test_loader)
    cc_total /= len(test_loader)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header (optional)
        writer.writerow(['MSE', 'R2', 'SNR', 'SNR_dB', 'RRMSE', 'RMSE', 'PCC', 'CC'])
        # Write rows
        for row in zip(mse_list, r2_list, snr_list, snr_db_list, rrmse_list, rmse_list, pcc_list, cc_list):
            writer.writerow(row)

    mse_list_total.extend(mse_list)
    r2_list_total.extend(r2_list)
    snr_db_list_total.extend(snr_db_list)
    pcc_list_total.extend(pcc_list)

    
    print(f'Data written to {csv_path}')

    print(
        "Average metrics: "
        f"MSE: {test_loss_total}, "
        f"R^2: {r2_metric_total}, "
        f"SNR in dB: {snr_db_total:.4f}, "
        f"PCC: {pcc_total:.4f}, "
    )

    metrics = {"MSE": test_loss_total, 
                "R^2": r2_metric_total,
                "SNR": snr_db_total, 
                "PCC": pcc_total,
                }

    df_met_per_sub[subject[0]] = pd.Series(metrics)  # Updates existing columns or adds new ones

    mse_list = [x for x in mse_list if math.isfinite(x)]
    r2_list = [x for x in r2_list if math.isfinite(x)]
    snr_list = [x for x in snr_list if math.isfinite(x)]

    print(f'Min MSE: {min(mse_list)}, Max MSE: {max(mse_list)}, Min R2: {min(r2_list)}, Max R2: {max(r2_list)}, Min SNR: {min(snr_list)}, Max SNR: {max(snr_list)},')


df_met_per_sub.to_csv(csv_summary_path)

metric_lists = {"MSE": mse_list_total, 
                "R^2": r2_list_total,
                "SNR": snr_db_list_total, 
                "PCC": pcc_list_total,
                }

results = []
for metric_name, values in metric_lists.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    results.append([metric_name, mean, std_dev])

# Write results to a CSV file

with open(final_summary_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Metric", "Mean", "Standard Deviation"])
    # Write metric data
    writer.writerows(results)

print(f"Summary saved to {final_summary_path}")