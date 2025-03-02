import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import math
import torch
from scipy import signal

from UPT4EEG.utils.misc import group_data_per_chan

def find_matching_string(lst, target):
    try:
        index = lst.index(target)  # This will return the index of the first match
        return index
    except ValueError:
        print('Given list does not include target string, hence -1 is returned.')
        return -1  # Return -1 if the target string is not found

""" pyplot waveform visualization """
def viewARA(tstmps, 
            tstmps_query, 
            data_colle, 
            ref_i, 
            electrode, 
            model_class = None,
            titles=None, 
            colors=None, 
            alphas=None, 
            ax=None, 
            picks_chs=None):
    n_data = len(data_colle)
    titles = ["" for di in range(n_data)] if titles is None else titles
    alphas = [0.5 for di in range(n_data)] if alphas is None else alphas
    if colors is None:
        cmap_ = plt.cm.get_cmap("tab20", n_data)
        colors = [rgb2hex(cmap_(di)) for di in range(n_data)]

    model_data_idx = find_matching_string(titles, target=model_class)
    
    picks = [electrode.index(c) for c in picks_chs]
    for di in range(n_data):
        data_colle[di] = data_colle[di][picks, :]
    if ax is None:
        ax = plt.subplot()
    for ii, ch_name in enumerate(picks_chs):
        offset = len(picks) - ii - 1
        norm_coef = 0.25 / np.abs(data_colle[ref_i][ii]).max()
        for di in range(n_data):
            if di == model_data_idx:
                eeg_dt = data_colle[di]
                ax.plot(tstmps_query, eeg_dt[ii] * norm_coef + offset,
                    label=None if ii else titles[di], color=colors[di], alpha=alphas[di],
                    linewidth=3 if alphas[di] > 0.6 else 1.5, # default=1.5
                )
            else:
                eeg_dt = data_colle[di]
                ax.plot(tstmps, eeg_dt[ii] * norm_coef + offset,
                    label=None if ii else titles[di], color=colors[di], alpha=alphas[di],
                    linewidth=3 if alphas[di] > 0.6 else 1.5, # default=1.5
                )                
    ax.set_xlim(tstmps[0], tstmps[-1])
    ax.set_ylim(-0.5, len(picks) - 0.5)

    ax.set_xlabel("Time (s)", fontsize=20)

    ax.set_xticks([])
    ax.set_xticks(np.arange(tstmps[0], tstmps[-1]+1, 1))
    ax.set_xticklabels(np.arange(int(tstmps[0]), int(tstmps[-1]+1), 1), fontsize=16)
    ax.set_yticks([])
    ax.set_yticks(np.arange(len(picks)))
    ax.set_yticklabels(picks_chs[::-1], fontsize=20)
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower right", borderaxespad=0, ncol=3, fontsize=20
    )

def ar_through_model(dataloader, 
                     model, 
                     query,
                     window_size, 
                     stride, 
                     sfreq,
                     device="cpu"
                     ):
    window_size_input = math.ceil(window_size * sfreq)
    stride_input = math.ceil(stride * sfreq)
    window_size_query = math.ceil(window_size * query['query_freq'])
    stride_query = math.ceil(stride * query['query_freq'])

    #eeg_data: a test batch dictionary
    model.eval()

    ch_nr_query = len(query['query_montage_pairs'])   #TODO we will need ch_nr_input as well if input and query ch_nr are different

    tstap_input = 0
    tstap_query = 0
    idx_tot_input = int(len(dataloader) * sfreq)
    idx_tot_query = int(len(dataloader) * query['query_freq'])

    target = []
    x_noisy = []

    noiseless_eeg = np.zeros((ch_nr_query, idx_tot_query), dtype=np.float32)
    x_noisy = np.zeros((ch_nr_query, idx_tot_input), dtype=np.float32)
    target = np.zeros((ch_nr_query, idx_tot_input), dtype=np.float32)
    
    hcoef = np.zeros((idx_tot_query), dtype=np.float32)
    hcoef_tar = np.zeros((idx_tot_input), dtype=np.float32)
    hcoef_x = np.zeros((idx_tot_input), dtype=np.float32)

    hwin = signal.windows.hann(window_size_query) + 1e-9
    hwin_tar = signal.windows.hann(window_size_input) + 1e-9
    hwin_x = signal.windows.hann(window_size_input) + 1e-9
    
    print(f'Number of segments in dataloader: {len(dataloader)}')
    for batch in dataloader:
        with torch.no_grad():
            pred_segment = model(
                input_feat=batch["input_feat"].to(device),
                input_pos=batch["input_pos"].to(device),
                batch_idx=batch["batch_idx"].to(device),
                output_pos=batch["query_pos"].to(device),
            )

        norm_factor = np.array(batch["norm_factor"].view(ch_nr_query, 1, 1))

        t = batch["query_pos"].squeeze()[:, -1]
        query_pos  = np.array(batch["query_pos"].squeeze()[:, 0:6])
        
        #print(f'query_pos.shape: {query_pos.shape}')

        t_tar = batch["target_pos"].squeeze()[:, -1]
        target_pos  = np.array(batch["target_pos"].squeeze()[:, 0:6])

        t_x = batch["input_pos"].squeeze()[:, -1]
        input_pos  = np.array(batch["input_pos"].squeeze()[:, 0:6])
        
        data_channel_dict, t_steps = group_data_per_chan(query_pos, pred_segment, t)
        data_channel_dict_tar, t_steps_tar = group_data_per_chan(target_pos, batch["target_feat"], t_tar)
        data_channel_dict_x, t_steps_x = group_data_per_chan(input_pos, batch["input_feat"], t_x)

        data_channel_dict = {k: v for k, v in data_channel_dict.items() if len(v) > 0}
        data_channel_dict_tar = {k: v for k, v in data_channel_dict_tar.items() if len(v) > 0}
        data_channel_dict_x = {k: v for k, v in data_channel_dict_x.items() if len(v) > 0}

        pred_segment = np.stack(list(data_channel_dict.values())) * norm_factor
        tar_segment = np.stack(list(data_channel_dict_tar.values())) * norm_factor
        x_segment = np.stack(list(data_channel_dict_x.values())) * norm_factor

        noiseless_eeg[:, tstap_query: tstap_query + window_size_query] += pred_segment.squeeze() * hwin
        hcoef[tstap_query: tstap_query + window_size_query] += hwin

        x_noisy[:, tstap_input: tstap_input + window_size_input] += x_segment.squeeze() * hwin_x
        hcoef_x[tstap_input: tstap_input + window_size_input] += hwin_x

        target[:, tstap_input: tstap_input + window_size_input] += tar_segment.squeeze() * hwin_tar
        hcoef_tar[tstap_input: tstap_input + window_size_input] += hwin_tar
        
        tstap_input += stride_input
        tstap_query += stride_query

        if tstap_query >= idx_tot_query-window_size_query+stride_query:
            break
            
    noiseless_eeg /= hcoef
    x_noisy /= hcoef_x
    target /= hcoef_tar
    

    return x_noisy, target, noiseless_eeg
