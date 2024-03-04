"""
+----------------------------------------------------------------------------------------------------------------------+
AMBULATORY SEIZURE FORECASTING USING WEARABLE DEVICES
Auxiliary functions for preprocessing and MSG data manipulation (preproc_utils.py)

Johannes Kepler University, Linz
ML Seminar/Practical Work 2023/24
Author:  Jozsef Kovacs
Created: 28/02/2024
+----------------------------------------------------------------------------------------------------------------------+
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.signal
import antropy as apy
from seerpy.utils import plot_eeg


def get_subject_path(data_dir, subject_id):
    return Path(data_dir) / subject_id


def get_metadata(data_dir, subject_id, folder, channel):
    meta_fn = subject_id + '_Empatica-' + folder + '_' + channel + '_metadata.csv'
    filepath = get_subject_path(data_dir, subject_id) / meta_fn
    return pd.read_csv(filepath)


def get_labels(data_dir, subject_id):
    labels_fn = subject_id + '_labels.csv'
    filepath = get_subject_path(data_dir, subject_id) / labels_fn
    return pd.read_csv(filepath)


def find_suitable_preictals(labels: pd.DataFrame, separation_th: float = 240.0, setback: float = 15.0, preictal_len: float = 60.0, verbose: bool = False):
    start_time = labels.loc[0, 'startTime']
    preictal_intervals = []

    for i in labels.index:
        curr_seizure_ts = labels.loc[i, 'labels.startTime']
        prev_seizure_ts = labels.loc[i-1, 'labels.startTime'] if i > 0 else start_time
        separation = (curr_seizure_ts - prev_seizure_ts) / 6e4
        if verbose:
            print('Seizure start:', datetime.fromtimestamp(curr_seizure_ts/1e3), f'(previous was {separation/60.0:.2f}h ago)', 'suitable:', separation>separation_th)
        if separation > separation_th:
            start_ts = labels.loc[i, 'labels.startTime'] - (setback + preictal_len) * 6e4
            preictal_intervals.append((start_ts, start_ts + preictal_len * 6e4 ))

    return preictal_intervals


def find_suitable_interictals(labels: pd.DataFrame, setback: float = 15.0, interval_len: float = 60.0, verbose: bool = False):
    start_time = labels.loc[0, 'startTime']
    duration = labels.loc[0, 'duration']

    seizure_times = [(labels.loc[i, 'labels.startTime'], labels.loc[i, 'labels.duration']) for i in labels.index]
    seizure_times.append((start_time + duration, 0))

    interictal_intervals = []

    i = 0
    start_ts = start_time
    while start_ts + interval_len * 6e4 < start_time + duration:
        end_ts = start_ts + interval_len * 6e4
        if start_ts + interval_len * 6e4 >= seizure_times[i][0] - (interval_len + setback) * 6e4:
            if verbose:
                print('<!NOT!>', [str(datetime.fromtimestamp(start_ts/1e3)) + " - " + str(datetime.fromtimestamp(end_ts/1e3))])
            start_ts = seizure_times[i][0] + seizure_times[i][1] + setback * 6e4
            i = min(i + 1, len(seizure_times) - 1)
        else:
            interictal_intervals.append((start_ts, end_ts))
            if verbose:
                print('< O K >', [str(datetime.fromtimestamp(start_ts/1e3)) + " - " + str(datetime.fromtimestamp(end_ts/1e3))])
            start_ts = end_ts

    return interictal_intervals


def retrieve_segment(data_dir, subject_id, folder, channel, segment_id):
    pq_folder = 'Empatica-' + folder
    pq_filename = subject_id + '_Empatica-' + folder + '_' + channel + '_segment_' + str(segment_id) + '.parquet'
    parquet_fn = get_subject_path(data_dir, subject_id) / pq_folder / pq_filename
    return pd.read_parquet(parquet_fn, engine='pyarrow')


def retrieve_segment_data(data_dir, subject_id, channel, metadata, start_ts: float, end_ts: float, verbose: bool = False):
    segments = metadata[channel][metadata[channel]['segments.startTime'] <= start_ts]
    if len(segments) == 0:
        return None

    segment_id = segments.index[-1]
    if verbose:
        print('start segment ID:', segment_id)
    segment_data = retrieve_segment(data_dir, subject_id, channel[0], channel[1], segment_id)

    while segment_data.iloc[-1, 0] <= end_ts:
        segment_id += 1
        segment_data = pd.concat([segment_data, retrieve_segment(data_dir, subject_id, channel[0], channel[1], segment_id)])

    result = segment_data[segment_data['time'].between(start_ts, end_ts)]
    if verbose:
        print('end segment ID:', segment_id, 'data len:', len(result))

    #return segment_data[segment_data['time'].between(start_ts, end_ts)].iloc[1, 0]
    return result


def get_input_data(data_dir, subject_id, metadata, channels, start_ts, end_ts):
    size = int((end_ts - start_ts) / 1000) * 128

    dfs = []
    min_ts = np.inf
    for folder, channels in channels.items():
        for ch in channels:
            df = retrieve_segment_data(data_dir, subject_id, (folder, ch), metadata, start_ts, end_ts)
            if df is None:
                return None
            df = df.rename(columns={"data": folder + "_" + ch})
            dfs.append(df)
            min_ts = min(df.iloc[0, 0], min_ts)

    res_df = pd.DataFrame([min_ts + i * 1000./128. for i in range(size)], columns=['time'])

    for df in dfs:
        res_df = pd.merge(res_df, df, on='time', how='outer')

    return res_df.iloc[:size, :]


def plot_data(df, start=0, length=1*60*128):
    end = start + length
    plot_data = df.iloc[start:end, 1]
    plot_data = plot_data - plot_data.median()
    plot_eeg(x=plot_data)


def plot_all_channels(df, start=0, length=1*60*128):
    end = start + length
    for i in range(1, 8):
        plot_data = df.iloc[start:end, i]
        plot_data = plot_data - plot_data.median()
        plot_eeg(x=plot_data)


def preproc_calculate_stft(df, freq=128, seqsize=4, overlap=3, step=5):
    df = df.fillna(0)
    data = np.transpose(df.values.tolist())
    f, t, Zxx = scipy.signal.stft(
        data,
        fs=freq,
        window="hann",
        nperseg=seqsize * freq,
        noverlap=overlap * freq,
        scaling="psd",
        return_onesided=True,
    )
    Zxx = Zxx[:, [i for i in range(0, f.shape[0], step)]]

    res = np.swapaxes(Zxx, 1, 2)
    res = np.swapaxes(res, 0, 1)

    res = np.reshape(res, (res.shape[0], -1))
    res = res.real.astype(np.float64)

    return res


def eda_sqi_calc(eda, window=128, artifact_lim=(-0.1, 0.2)):
    amp = np.mean(eda[0:window])
    res = np.zeros(eda.shape, dtype=np.float64)
    for i in range(0, eda.shape[0], window):
        rate = (np.mean(eda[i:i+window]) - amp) / amp
        res[i:i + window + 1] = min(max(rate, artifact_lim[0]), artifact_lim[1])
    return res


def bvp_sqi_spectral_entropy(bvp, freq=128, window=4, segsize=60):
    res = np.zeros(bvp.shape, dtype=np.float64)
    for j in range(0, bvp.shape[0], segsize * freq):
        res[j:j + segsize * freq + 1] = np.mean([apy.spectral_entropy(bvp[i: i + window * freq + 1], sf=freq, method='welch', normalize=True) for i in range(j, j + segsize*freq, window * freq)])
    return res


def accmag_sqi_spower_ratio(accmag, freq=128, window=4, segsize=60, physio=(.8, 5.), broadband=(.8, )):
    res = np.zeros(accmag.shape, dtype=np.float64)
    for j in range(0, accmag.shape[0], segsize * freq):
        ratio = []
        for i in range(j, j + segsize*freq, window * freq):
            fft_accmag = np.fft.rfft(accmag[i: i + window * freq])
            freq_accmag = np.fft.rfftfreq(len(fft_accmag), 1./freq)
            narrow_physio_domain = np.sum([np.abs(v) for f, v in zip(freq_accmag, fft_accmag) if physio[0] <= f <= physio[1]])
            broadband_domain = np.sum([np.abs(v) for f, v in zip(freq_accmag, fft_accmag) if broadband[0] <= f])
            ratio.append(narrow_physio_domain / (broadband_domain + 1e-15))
        res[j:j + segsize * freq + 1] = np.mean(ratio)

    return res


def preprocess(df: pd.DataFrame):
    # preprocess: calculate ShortTime FFT for selected channels
    x = df[['BVP_BVP', 'TEMP_TEMP', 'EDA_EDA', 'HR_HR', 'ACC_Acc Mag']]
    tr = preproc_calculate_stft(x, freq=128, seqsize=4, overlap=3, step=5)

    # preprocess: calculate SQI for the EDA channel - rate of change in amplitude
    eda_sqi = eda_sqi_calc(np.asarray(df[['EDA_EDA']]).reshape(-1,))

    # preprocess: calculate SQI for the BVP channel - spectral entropy
    bvp_sqi = bvp_sqi_spectral_entropy(np.asarray(df[['BVP_BVP']]).reshape(-1,), window=4, segsize=60)

    # preprocess: calculate SQU for the ACC Mag channel - spectral power ratio
    accmag_sqi = accmag_sqi_spower_ratio(np.asarray(df[['ACC_Acc Mag']]).reshape(-1), freq=128, window=4, segsize=60, physio=(.8, 5.), broadband=(.8, 64.))

    # preprocess: add 24-hour part of the timestamp
    h24_info = np.asarray([datetime.fromtimestamp(ts/1000.0).hour for ts in df[['time']].iloc[:,0]])

    # concatenate input
    features = np.hstack([np.asarray(df)[:, 1:], np.repeat(tr, 128, axis=0)[:x.shape[0]], eda_sqi.reshape(-1, 1), bvp_sqi.reshape(-1, 1), accmag_sqi.reshape(-1, 1), h24_info.reshape(-1, 1)])

    return features


def calc_z_score(data_dir, subject_id, metadata, intervals, feature_dim):
    M, SD, N = np.zeros((feature_dim,), dtype=np.float64), np.zeros((feature_dim,), dtype=np.float64), 0
    for iv in intervals:
        start_ts, end_ts = iv
        inp_data = get_input_data(data_dir, subject_id, metadata, start_ts, end_ts)
        if inp_data is None:
            continue
        data = preprocess(inp_data)
        M += np.sum(data, axis=0)
        SD += np.sum(data**2, axis=0)
        N += data.shape[0]

    means = np.nan_to_num(M / N, nan=0.)
    stds = np.nan_to_num((np.sqrt((SD/N) - (M/N)**2)), nan=1.)

    return means, stds


def normalize(data, means, stds):
    return (data - means) / stds
