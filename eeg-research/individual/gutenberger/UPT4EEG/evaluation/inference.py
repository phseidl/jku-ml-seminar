# check successful setup
import torch
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

import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

from torch import from_numpy as np2TT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pathlib import Path
import argparse
import pandas as pd


from UPT4EEG.utils.build import create_dataset
from UPT4EEG.evaluation.eval_metrics import calc_eval_metrics
from UPT4EEG.dataset.sparse_eeg_dataset_val import SparseEEGDataset_val
from UPT4EEG.dataset.collator_val import SparseEEG_Collator
from UPT4EEG_mp_2.model.encoder import Encoder
from UPT4EEG_mp_2.model.decoder import DecoderPerceiver
from UPT4EEG_mp_2.model.UPT4EEG import UPT4EEG

# Create the parser
parser = argparse.ArgumentParser(description="Inference script to generate eval metrics for UPT4EEG.")

# Define the argument
parser.add_argument('model_type', type=str, help="Which model to choose. Either: ['big_rand', 'small_tuh', 'small_tuh_same', 'small_rdm_same', 'small_rdm_cd', 'small_rd', 'CLEAN-E1', 'CLEAN-E2]")
parser.add_argument('use_montage', type=str, help="Defines which montage to use in inference ('tuh', 'tuh_rand' or 'random').")

# Parse the arguments
args = parser.parse_args()

saved_model_type = args.model_type
use_montage = args.use_montage

DATASET = 'TUH'    # either 'TUH' or 'BCI' or 'DenoiseNet'
MODEL_CLASS = 'UPT4EEG'
plot_sample = False

config_path = '/system/user/studentwork/gutenber/configs/config.yml'

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())


SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
#use_montage = cfg_dataset['use_montage']
#use_montage = 'user_specific'
#use_montage = 'tuh'
#NUM_EPOCHS = 1 #cfg_general['epochs']
#BATCH_SIZE = cfg_model['batch_size']
#LR         = cfg_model["learning_rate"]


SAVE_PATH = 'logs/' + DATASET + '/' + MODEL_CLASS

if not os.path.exists(SAVE_PATH):
    try:
        os.makedirs(SAVE_PATH)
    except Exception as e:
        print(f"Failed to create directory '{SAVE_PATH}': {e}")


timestamp = datetime.now().strftime("%b%d_%H-%M-%S")


window_size = 1
stride = 0.5

df_met_per_sub = pd.DataFrame()
mse_list_total = []
r2_list_total = []
snr_db_list_total = []
pcc_list_total = []


for subject in cfg_dataset["subjects_test"]:

    x_test, y_test, ch_names = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        [subject], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
    )

    #cfg_dataset["window_size"]
    #x_train.shape: [Nr of segments, channel nr, sequence length]
    x_test = np2TT(x_test)
    y_test = np2TT(y_test)


    #model_path = '/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Jan02_14-31-42_train.pth' #denoising trained on tuh
    #model_path = '/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Jan07_11-58-55_val.pth' #denoising trained on random, val on tuh
    if saved_model_type == 'big_rand':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan18_10-47-07_val.pth' 
        d_model = 192*4
        dim = 256   
        num_heads = 8 #3
        depth = 6
    #model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_17-49-57_val.pth'  #trained with bin loss!!
    elif saved_model_type == 'small_tuh':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_18-47-03_val.pth'  #small model on tuh
        d_model = 192*2
        dim = 192  
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_tuh_same':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-21-01_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm_cd_same':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-11-27_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm_cd':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-12-03_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-12-29_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_tuh_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_18-47-03_train.pth'  #small model on tuh
        d_model = 192*2
        dim = 192  
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_tuh_same_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-21-01_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm_cd_same_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-11-27_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm_cd_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-12-03_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'small_rdm_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan23_19-12-29_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'ensemble_loss':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan20_17-12-18_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'ensemble_loss_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan20_17-12-18_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'bin_loss_tuh':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan30_15-12-47_val.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'bin_loss_tuh_train':
        model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan30_15-12-47_train.pth'
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
    elif saved_model_type == 'CLEAN-E1' or saved_model_type == 'CLEAN-E2' or saved_model_type == 'CLEAN-E3' or saved_model_type == 'CLEAN-E4':
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3
        if saved_model_type == 'CLEAN-E1':
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan31_15-49-07_val.pth'
        if saved_model_type == 'CLEAN-E2':
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Feb16_12-45-17_val.pth'
        if saved_model_type == 'CLEAN-E3':
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Feb16_12-46-16_val.pth'
        elif saved_model_type == 'CLEAN-E4':
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan31_15-50-05_val.pth'
    elif saved_model_type == 'CLEAN-B0' or saved_model_type == 'CLEAN-B1' or saved_model_type == 'CLEAN-B2' or saved_model_type == 'CLEAN-B3' or saved_model_type == 'CLEAN-B4':
        d_model = 192*2
        dim = 192   
        num_heads = 4 #3
        depth = 3        
        if saved_model_type == 'CLEAN-B0':  
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan30_15-12-47_val.pth'
        if saved_model_type == 'CLEAN-B1':  
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Feb01_14-34-51_val.pth'
        if saved_model_type == 'CLEAN-B2':  
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan31_09-29-55_val.pth'
        if saved_model_type == 'CLEAN-B3':  
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan31_15-47-14_val.pth'
        if saved_model_type == 'CLEAN-B4':  
            model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan31_15-38-24_val.pth'


    #model_path = '/system/user/studentwork/gutenber/upt-minimal/upt-minimal/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Dec30_16-14-33_val.pth' #recostruction
    #'/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Dec22_10-45-06_val.pth'
    else:
        raise Exception("Variable saved_model_type invalid.")
    # hyperparameters


    num_supernodes = 512
    input_dim = 1
    output_dim = 1
    use_mlp_posEnc = True



    # initialize model
    model = UPT4EEG(
        encoder = Encoder(
            input_dim=input_dim,
            ndim=1,
            gnn_dim=d_model,
            enc_dim=dim,
            enc_num_heads=num_heads,
            enc_depth=depth,
            mlp_pos_enc=use_mlp_posEnc,
        ),
        decoder=DecoderPerceiver(
            input_dim=dim,
            output_dim=output_dim,
            ndim=1,
            dim=dim,
            num_heads=num_heads,
            depth=depth,
            mlp_pos_enc=use_mlp_posEnc,
        ),
    )

    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


    state_path = os.path.join(model_path)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])

    query_montage_pairs = [
                ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
                ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
                ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
                ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
                ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
                ('C4', 'P4'), ('P4', 'O2')
                ]
    query = {'query_freq': 3000.0, 'query_montage_pairs': query_montage_pairs}


    test_dataset = SparseEEGDataset_val(x_test, y_test, query, ch_names, cfg_dataset, use_montage=use_montage)

    sample = test_dataset[0]
    print(f"Input features shape: {sample['input_feat'].shape}")
    print(f"Input positions shape: {sample['input_pos'].shape}")
    print(f"Target features shape: {sample['target_feat'].shape}")
    print(f"Output positions shape: {sample['target_pos'].shape}")
    print(f"Query positions shape: {sample['query_pos'].shape}")

    # setup dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        collate_fn=SparseEEG_Collator(num_supernodes=256, deterministic=False),
    )


    ################################## EVALUATION METRICS ##########################
    csv_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/eval_metrics_upt4eeg_' + saved_model_type + '_montage_' + use_montage + '.csv'
    csv_summary_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/subject_metrics_summary_upt4eeg_' + saved_model_type + '_montage_' + use_montage + '.csv'
    final_summary_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/metrics_summary_upt4eeg_'+ saved_model_type + '_montage_' + use_montage + '.csv'

    total_updates = len(test_dataloader)

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
    #pbar.set_description(f"MSE loss: {test_loss_total:.4f}, SNR in dB: {snr_total:.4f}, R^2 score: {r2_metric_total:.4f}, PCC: {pcc_total:.4f}, RRMSE: {rrmse_total:.4f}")
        

    for batch in test_dataloader:
        with torch.no_grad():
            y_hat = model(
                input_feat=batch["input_feat"].to(device),
                input_pos=batch["input_pos"].to(device),
                batch_idx=batch["batch_idx"].to(device),
                output_pos=batch["target_pos"].to(device),
            )
        y = batch["target_feat"].to(device)
        eval_metrics = calc_eval_metrics(y_hat, y)
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
        
    test_loss_total /= len(test_dataloader)
    r2_metric_total /= len(test_dataloader)
    snr_total /= (len(test_dataloader))
    snr_db_total /= (len(test_dataloader))
    rrmse_total /= len(test_dataloader)
    rmse_total /= len(test_dataloader)
    pcc_total /= len(test_dataloader)
    cc_total /= len(test_dataloader)

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