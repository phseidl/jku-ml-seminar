import numpy as np
from sklearn.metrics import r2_score
import torch.nn.functional as F
import math
from torch import from_numpy as np2TT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import yaml
import os
from pathlib import Path
import torch

from BaselineModels.utils.build import create_dataset

def calc_eval_metrics(y_hat, y):
    y = y.squeeze().flatten()
    y_hat = y_hat.squeeze().flatten()
    mse = (F.mse_loss(y_hat, y)).item()
    r2 = r2_score(np.array(y_hat.cpu()), np.array(y.cpu()))
    snr_db = SNR(np.array(y.cpu()), np.array(y_hat.cpu()), inDezibel=True)
    snr = SNR(np.array(y.cpu()), np.array(y_hat.cpu()), inDezibel=False)
    rmse = RMSE(np.array(y.cpu()), np.array(y_hat.cpu()))
    rrmse = RRMSE(np.array(y.cpu()), np.array(y_hat.cpu()))
    cc = CC(np.array(y.cpu()), np.array(y_hat.cpu()))
    pcc = PCC(np.array(y.cpu()), np.array(y_hat.cpu()))
    eval_metrics = {
        "MSE": mse,
        "R2": r2,
        "SNR_dB": snr_db,
        "SNR": snr,
        "RRMSE": rrmse,
        "CC": cc,
        "PCC": pcc,
        "RMSE": rmse,
    }
    return eval_metrics

def SNR(clean_data, noisy_data, inDezibel = True):
    # clean data: reference data
    # noisy data: data to measure SNR on, e.g. output of the model
    clean_data = clean_data.flatten().squeeze()
    noisy_data = noisy_data.flatten().squeeze()

    if inDezibel:
        return 10 * np.log10(np.sum(clean_data**2)/np.sum((clean_data-noisy_data)**2))
    else:
        return np.sum(clean_data**2)/np.sum((clean_data-noisy_data)**2)

def MSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def RMSE(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def RRMSE(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_true = np.mean(y_true)
    rrmse = rmse / abs(mean_true)
    return rrmse

def CC(y_true, y_pred):
    numerator = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
    denominator = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2) * np.sum((y_pred - np.mean(y_pred)) ** 2))
    correlation = numerator / denominator
    return correlation

def PCC(y_true, y_pred):
    pearson_corr = np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]
    return pearson_corr






if __name__ == "__main__":
    DATASET = 'TUH'
    workin_dir = '/system/user/studentwork/gutenber'

    config_path = os.path.join(workin_dir, 'configs/config.yml')
    model_config_path = os.path.join(workin_dir, 'configs/model_config.yml')

    model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
    cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
    cfg_general = yaml.safe_load(Path(config_path).read_text())

    SFREQ      = cfg_dataset["sfreq"]
    normalize  = cfg_dataset["normalize"]

    use_montage = 'tuh'

    window_size = 4
    stride = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_test, y_test = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        cfg_dataset["subjects_test"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
    )

    x_test = np2TT(np.expand_dims(x_test, axis=1))
    y_test = np2TT(np.expand_dims(y_test, axis=1))

    testset = torch.utils.data.TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=True)

    csv_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/eval_metrics_noisy.csv'

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
        

    for x, y in test_dataloader:
        eval_metrics = calc_eval_metrics(x, y)
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

            

    print(f'Data written to {csv_path}')

    print(
        f"AVG. MSE TEST LOSS: {test_loss_total}, "
        f"AVG. R^2 TEST LOSS: {r2_metric_total}, "
        f"AVG. SNR: {snr_total:.4f}, "
        f"AVG. SNR in dB: {snr_db_total:.4f}, "
        f"AVG. PCC: {pcc_total:.4f}, "
        f"AVG. RRMSE: {rrmse_total:.4f}, "
        f"AVG. RMSE: {rmse_total:.4f}"
    )


    mse_list = [x for x in mse_list if math.isfinite(x)]
    r2_list = [x for x in r2_list if math.isfinite(x)]
    snr_list = [x for x in snr_list if math.isfinite(x)]

    print(f'Min MSE: {min(mse_list)}, Max MSE: {max(mse_list)}, Min R2: {min(r2_list)}, Max R2: {max(r2_list)}, Min SNR: {min(snr_list)}, Max SNR: {max(snr_list)},')
