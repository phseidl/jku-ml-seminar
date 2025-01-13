from utils.cleegn import CLEEGN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy as np2TT
from torchinfo import summary
from omegaconf import OmegaConf
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
from dacite import from_dict
from dacite import Config as DaciteConfig
import numpy as np
import math
import json
import time
import sys
import os
import yaml
from pathlib import Path

from utils.seq2seq import Seq2Seq, Seq2SeqLSTM, LSTM
from utils.seq2seq_attention import Seq2SeqWithAttention
from utils.lstm_autoencoder import LSTMAutoencoder
from main import model_select
from utils.build import get_rdm_EEG_segment_DenoiseNet


MODEL_CLASS = 'OneD_Res_CNN'
DATASET = 'DenoiseNet'
MODEL_FILE_NAME = 'OneD_Res_CNN_Nov13_13-35-41.pth'
artifact_type = 'EOG'
snr_synthetic_testData = 4 #in dezibel
plt_interval = [0, 512]


if DATASET == 'BCI':
    config_path = 'configs/BCI_KAGGLE/config.yml'
    model_config_path = 'configs/BCI_KAGGLE/model_config.yml'
    electrode = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'PO8', 'O1', 'O2']
    picks_chs = ["Fp1", "Fp2", "F7", "T7", "O2"]
    TEST_DATA_PATH = 'sampleData\Data_S14.mat'
    
elif DATASET == 'TUH':
    config_path = 'configs/TUH/config.yml'
    model_config_path = 'configs/TUH/model_config.yml'
    electrode = ['FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
    picks_chs = ["FP1", "FP2", "F7", "T4", "O2"]
    TEST_DATA_PATH = 'data\TUH_TUSZ\TUH_dataset_inference\Data_S015_norm.mat'

elif DATASET == 'DenoiseNet':
    config_path = 'configs/EEG_DenoiseNet/config.yml'
    model_config_path = 'configs/EEG_DenoiseNet/model_config.yml'
    electrode = ['ch1']
    picks_chs = ['ch1']


cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
SFREQ      = cfg_dataset["sfreq"]


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)




torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
xlstm_cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=3
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_heads=1,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=512,
    num_blocks=1,
    embedding_dim=18,
    slstm_at=[0],
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" pyplot waveform visualization """
def viewARA(tstmps, data_colle, ref_i, electrode, titles=None, colors=None, alphas=None, ax=None, picks_channel=None):
    n_data = len(data_colle)
    titles = ["" for di in range(n_data)] if titles is None else titles
    alphas = [0.5 for di in range(n_data)] if alphas is None else alphas
    if colors is None:
        cmap_ = plt.cm.get_cmap("tab20", n_data)
        colors = [rgb2hex(cmap_(di)) for di in range(n_data)]

    
    picks = [electrode.index(c) for c in picks_chs]
    for di in range(n_data):
        data_colle[di] = data_colle[di][picks, :]
    if ax is None:
        ax = plt.subplot()
    for ii, ch_name in enumerate(picks_chs):
        offset = len(picks) - ii - 1
        norm_coef = 0.25 / np.abs(data_colle[ref_i][ii]).max()
        for di in range(n_data):
            eeg_dt = data_colle[di]
            ax.plot(tstmps, eeg_dt[ii] * norm_coef + offset,
                label=None if ii else titles[di], color=colors[di], alpha=alphas[di],
                linewidth=3 if alphas[di] > 0.6 else 1.5, # default=1.5
            )
    ax.set_xlim(tstmps[0], tstmps[-1])
    ax.set_ylim(-0.5, len(picks) - 0.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticks(np.arange(len(picks)))
    ax.set_yticklabels(picks_chs[::-1], fontsize=20)
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower right", borderaxespad=0, ncol=3, fontsize=20
    )

def ar_through_model(eeg_data, model, window_size, stride):
    model.eval()

    noiseless_eeg = np.zeros(eeg_data.shape, dtype=np.float32)
    hcoef = np.zeros(eeg_data.shape[1], dtype=np.float32)

    hwin = signal.windows.hann(window_size) + 1e-9
    for i in range(0, noiseless_eeg.shape[1], stride):
        tstap, LAST_FRAME = i, False
        segment = eeg_data[:, tstap: tstap + window_size]
        if segment.shape[1] != window_size:
            tstap = noiseless_eeg.shape[1] - window_size
            segment = eeg_data[:, tstap:]
            LAST_FRAME = True
        with torch.no_grad():
            segment = np.expand_dims(segment, axis=0)
            data = np2TT(np.expand_dims(segment, axis=0))
            if MODEL_CLASS == 'xLSTM':
                data = data.permute(0,1,3,2).squeeze(0)  #ADDED
            data = data.to(device, dtype=torch.float)
            pred_segment = model(data)
            if MODEL_CLASS == 'xLSTM':
                pred_segment = pred_segment.permute(0,2,1)
            pred_segment = np.array(pred_segment.cpu()).astype(np.float32)   #pred_segment [1, n_chan, seq_length]
        noiseless_eeg[:, tstap: tstap + window_size] += pred_segment.squeeze() * hwin
        hcoef[tstap: tstap + window_size] += hwin

        if LAST_FRAME:
            break
    noiseless_eeg /= hcoef

    return noiseless_eeg

def calc_SNR(clean_data, noisy_data, inDezibel = True):
    # clean data: reference data
    # noisy data: data to measure SNR on, e.g. output of the model
    n_chan = clean_data.shape[0]

    if inDezibel:
        return 1/n_chan * np.sum(10 * np.log10(np.linalg.norm(clean_data, axis = 1)/np.linalg.norm(clean_data-noisy_data, axis = 1)))
    else:
        return 1/n_chan * np.sum(np.linalg.norm(clean_data, axis = 1)**2/np.linalg.norm(clean_data-noisy_data, axis = 1)**2)

def calc_MSE(x, y):
    return 1/x.shape[0] * np.sum(1/x.shape[1] * np.linalg.norm(x - y, axis = 1)**2)


if __name__ == "__main__":
    import argparse

    model_path = os.path.join(os.path.abspath(os.getcwd()), 'logs', DATASET, MODEL_CLASS, MODEL_FILE_NAME)
    
    if DATASET == 'TUH' or DATASET == 'BCI':
        test_data = loadmat(TEST_DATA_PATH)
        noisy_data, reference_data = test_data["x_test"], test_data["y_test"]
    elif DATASET == 'DenoiseNet':
        noisy_data, reference_data = get_rdm_EEG_segment_DenoiseNet(cfg_dataset, artifact_type, snr_synthetic_testData)
        percentile_95 = np.quantile(np.abs(noisy_data.squeeze()), 0.95) 
        noisy_data = noisy_data/percentile_95 
        reference_data = reference_data/percentile_95


    state_path = os.path.join(model_path)
    state = torch.load(state_path, map_location="cpu")

    #xlstm_stack = xLSTMBlockStack(xlstm_cfg)

    model = model_select(MODEL_CLASS, cfg_model)
    model.load_state_dict(state["state_dict"])

    reconstructed_data = ar_through_model(
        noisy_data, model, math.ceil(4.0 * 128.0), math.ceil(1 * 128.0)
    )
    


    start = plt_interval[0]
    x_min, x_max = start, start + plt_interval[1]
    x_data = noisy_data[:, x_min: x_max]
    y_data = reference_data[:, x_min: x_max]
    p_data = reconstructed_data[:, x_min: x_max]

    #TODO SNR, MSE berechnen und printen
    snr = calc_SNR(y_data, p_data, inDezibel=False)
    snr_dB = calc_SNR(y_data, p_data, inDezibel=True)
    mse = calc_MSE(y_data, p_data)
    
    print('Data points of segment: ' + str(p_data.shape[1]))
    print(f'MSE: {mse:.5f}')
    print(f'SNR: {snr_dB:.2f}dB (or {snr:.2f})')

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    viewARA(
        np.linspace(0, math.ceil(x_data.shape[-1] / 128.0), x_data.shape[-1]),
        [x_data, y_data, y_data, p_data], 1, electrode,
        titles=["Original", "", "Reference", MODEL_CLASS], colors=["gray", "gray", "red", "blue"], alphas=[0.5, 0, 0.8, 0.8], ax=ax,
        picks_channel = picks_chs
    )
    plt.savefig("inference.pdf", format="pdf", bbox_inches="tight")
    plt.show()
