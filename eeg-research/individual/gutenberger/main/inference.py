from utils.cleegn import CLEEGN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy as np2TT
from torchinfo import summary

from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
import numpy as np
import math
import json
import time
import sys
import os

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
            num_heads=3,
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

MODEL_CLASS = 'xLSTM'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#electrode = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'PO8', 'O1', 'O2']
electrode = ['FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
""" pyplot waveform visualization """
def viewARA(tstmps, data_colle, ref_i, electrode, titles=None, colors=None, alphas=None, ax=None):
    n_data = len(data_colle)
    titles = ["" for di in range(n_data)] if titles is None else titles
    alphas = [0.5 for di in range(n_data)] if alphas is None else alphas
    if colors is None:
        cmap_ = plt.cm.get_cmap("tab20", n_data)
        colors = [rgb2hex(cmap_(di)) for di in range(n_data)]

    picks_chs = ["FP1", "FP2", "F7", "T4", "O2"]
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

    hwin = signal.hann(window_size) + 1e-9
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
            pred_segment = pred_segment.permute(0,2,1)
            pred_segment = np.array(pred_segment.cpu()).astype(np.float32)
        noiseless_eeg[:, tstap: tstap + window_size] += pred_segment.squeeze() * hwin
        hcoef[tstap: tstap + window_size] += hwin

        if LAST_FRAME:
            break
    noiseless_eeg /= hcoef

    return noiseless_eeg

if __name__ == "__main__":
    import argparse

    test_data_path = 'sampleData/Data_S016_norm.mat'

    model_path = 'logs/TUH/xLSTM/cleegn_tuh_xLSTM.pth'
    
    test_data = loadmat(test_data_path)
    dt_polluted, dt_ref = test_data["x_test"], test_data["y_test"]

    ### temporary fixed mode
    state_path = os.path.join(model_path)
    state = torch.load(state_path, map_location="cpu")

    xlstm_stack = xLSTMBlockStack(xlstm_cfg)
    #model = CLEEGN(n_chan=x_train.size()[2], fs=SFREQ, N_F=x_train.size()[2]).to(device)
    model = xlstm_stack.to(device)
    model.load_state_dict(state["state_dict"])
    #model.load_state_dict(torch.load(model_path))
    dt_cleegn = ar_through_model(
        dt_polluted, model, math.ceil(4.0 * 128.0), math.ceil(1 * 128.0)
    )
    
    start = 1500
    x_min, x_max = start, start + 500
    x_data = dt_polluted[:, x_min: x_max]
    y_data = dt_ref[:, x_min: x_max]
    p_data = dt_cleegn[:, x_min: x_max]
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    viewARA(
        np.linspace(0, math.ceil(x_data.shape[-1] / 128.0), x_data.shape[-1]),
        [x_data, y_data, y_data, p_data], 1, electrode,
        titles=["Original", "", "Reference", "CLEEGN"], colors=["gray", "gray", "red", "blue"], alphas=[0.5, 0, 0.8, 0.8], ax=ax
    )
    plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    plt.show()

