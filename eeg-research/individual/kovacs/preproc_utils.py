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
from utils import console_msg

from msg_subject_data import MsgSubjectData
from sklearn.preprocessing import StandardScaler
from skimage.measure import block_reduce
from scipy.signal import butter, filtfilt


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


def find_suitable_preictals(labels: pd.DataFrame, max_common_timestamp, separation_th: float = 240.0, setback: float = 15.0, preictal_len: float = 60.0, verbose: bool = False):
    start_time = labels.loc[0, 'startTime']
    preictal_intervals = []

    for i in labels.index:
        curr_seizure_ts = labels.loc[i, 'labels.startTime']
        if curr_seizure_ts >= max_common_timestamp:
            break
        prev_seizure_ts = labels.loc[i-1, 'labels.startTime'] if i > 0 else start_time
        separation = (curr_seizure_ts - prev_seizure_ts) / 6e4
        if verbose:
            print('Seizure start:', datetime.fromtimestamp(curr_seizure_ts/1e3), f'(previous was {separation/60.0:.2f}h ago)', 'suitable:', separation > separation_th)
        if separation > separation_th or (i == 0 and labels.loc[i, 'labels.startTime'] - (setback + preictal_len) * 6e4 >= start_time):
            start_ts = labels.loc[i, 'labels.startTime'] - (setback + preictal_len) * 6e4
            preictal_intervals.append((start_ts, start_ts + preictal_len * 6e4))

    return preictal_intervals


def find_suitable_interictals(msd: MsgSubjectData, labels: pd.DataFrame, max_common_timestamp, separation: float = 240.0, interval_len: float = 60.0, verbose: bool = False):
    start_time = labels.loc[0, 'startTime']
    duration = labels.loc[0, 'duration']

    seizure_times = [(labels.loc[i, 'labels.startTime'], labels.loc[i, 'labels.duration']) for i in labels.index if labels.loc[i, 'labels.startTime'] < max_common_timestamp]
    seizure_times.append((start_time + duration, 0))

    interictal_intervals = []

    i = 0
    start_ts = start_time
    while start_ts + interval_len * 6e4 < start_time + duration:
        end_ts = start_ts + interval_len * 6e4
        if start_ts + interval_len * 6e4 >= seizure_times[i][0] - (interval_len + separation) * 6e4:
            if verbose:
                print('<!NOT!>', [str(datetime.fromtimestamp(start_ts/1e3)) + " - " + str(datetime.fromtimestamp(end_ts/1e3))])
            start_ts = seizure_times[i][0] + seizure_times[i][1] + separation * 6e4
            i = min(i + 1, len(seizure_times) - 1)
        else:
            df = get_input_data(msd.mdh.input_dir, msd.subject_id, msd.metadata, msd.mdh.available_channels, start_ts, end_ts)
            if df is None or len(df) == 0:
                start_ts = end_ts
                continue
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

    while segment_data.iloc[-1, 0] <= end_ts and segment_id < len(metadata[channel]) - 1:
        segment_id += 1
        segment_data = pd.concat([segment_data, retrieve_segment(data_dir, subject_id, channel[0], channel[1], segment_id)])

    result = segment_data[segment_data['time'].between(start_ts, end_ts)]
    if verbose:
        print('end segment ID:', segment_id, 'data len:', len(result))

    #return segment_data[segment_data['time'].between(start_ts, end_ts)].iloc[1, 0]
    return result


def get_input_data(data_dir, subject_id, metadata, available_channels, start_ts, end_ts):
    size = int((end_ts - start_ts) / 1000) * 128

    dfs = []
    min_ts = np.inf
    for folder, channels in available_channels.items():
        for ch in channels:
            df = retrieve_segment_data(data_dir, subject_id, (folder, ch), metadata, start_ts, end_ts)
            if df is None or len(df) == 0:
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


def preproc_calculate_stft(df, freq=128, seg_size=4, overlap=3, freq_pool=16, time_pool=1):
    df = df.fillna(0)
    data = np.transpose(df.values.tolist())
    f, t, Zxx = scipy.signal.stft(
        data,
        fs=freq,
        window="hann",
        nperseg=seg_size,
        noverlap=overlap,
        scaling="psd",
        return_onesided=True,
        boundary=None
    )

    rZxx = block_reduce(np.abs(Zxx), block_size=(1, freq_pool, time_pool), func=np.max)

    res = np.swapaxes(rZxx, 1, 2)
    res = np.swapaxes(res, 0, 1)

    res = np.reshape(res, (res.shape[0], -1))

    # pad if necessary
    exp_shape = data.shape[1] // freq
    if res.shape[0] < exp_shape:
        res = np.concatenate([res, res[-(exp_shape-res.shape[0]):]], axis=0)

    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(4096), data[0, :4096])
    # plt.show()
    # rf = f[0::8]
    # tf = t[0::1]
    # plt.pcolormesh(tf, rf, rZxx, vmin=0, vmax=0.5)

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


def preprocess(config, df: pd.DataFrame):

    # preprocess: calculate short-time FFT for selected channels
    x = df[['BVP_BVP', 'TEMP_TEMP', 'EDA_EDA', 'HR_HR', 'ACC_Acc Mag']]
    tr = preproc_calculate_stft(x, freq=config.stft.freq, seg_size=config.stft.seg_size,
                                overlap=config.stft.overlap, freq_pool=config.stft.freq_pool,
                                time_pool=config.stft.time_pool)

    # preprocess: calculate SQI for the EDA channel - rate of change in amplitude
    eda_sqi = eda_sqi_calc(np.asarray(df[['EDA_EDA']]).reshape(-1,))

    # preprocess: calculate SQI for the BVP channel - spectral entropy
    bvp_sqi = bvp_sqi_spectral_entropy(np.asarray(df[['BVP_BVP']]).reshape(-1,), window=4, segsize=60)

    # preprocess: calculate SQU for the ACC Mag channel - spectral power ratio
    accmag_sqi = accmag_sqi_spower_ratio(np.asarray(df[['ACC_Acc Mag']]).reshape(-1), freq=128, window=4, segsize=60, physio=(.8, 5.), broadband=(.8, 64.))

    # preprocess: add 24-hour part of the timestamp
    h24_info = np.asarray([datetime.fromtimestamp(ts/1000.0).hour for ts in df[['time']].iloc[:,0]])

    # concatenate input
    data = np.asarray(df)[:, 1:]
    sqi = np.hstack([eda_sqi.reshape(-1, 1), bvp_sqi.reshape(-1, 1), accmag_sqi.reshape(-1, 1)])
    avg_dim = config.stft.seg_size - config.stft.overlap
    if config.stft.reduce:
        data = np.mean(data.reshape(-1, avg_dim, data.shape[-1]), axis=1)
        sqi = np.mean(sqi.reshape(-1, avg_dim, sqi.shape[-1]), axis=1)
        ft = tr[:data.shape[0]]
    else:
        ft = np.repeat(tr, avg_dim, axis=0)[:data.shape[0]]

    features = np.hstack([data, ft, sqi])
    features = np.nan_to_num(features)

    return features


def create_std_scaler(data_dir, subject_id, metadata, available_channels, intervals):
    scaler = StandardScaler()
    for iv in intervals:
        start_ts, end_ts = iv
        inp_data = get_input_data(data_dir, subject_id, metadata, available_channels, start_ts, end_ts)
        if inp_data is None or len(inp_data) == 0:
            continue
        d = np.nan_to_num(inp_data)
        scaler.partial_fit(d)
    return scaler


def calc_z_score(config, data_dir, subject_id, metadata, available_channels, intervals, feature_dim):
    M, SD, N = np.zeros((feature_dim,), dtype=np.float64), np.zeros((feature_dim,), dtype=np.float64), 0
    for iv in intervals:
        start_ts, end_ts = iv
        inp_data = get_input_data(data_dir, subject_id, metadata, available_channels, start_ts, end_ts)
        if inp_data is None or len(inp_data) == 0:
            continue
        data = preprocess(config, inp_data)
        data = np.nan_to_num(data)
        M += np.sum(data, axis=0)
        SD += np.sum(data**2, axis=0)
        N += data.shape[0]

    means = np.nan_to_num(M / N, nan=0.)
    stds = np.nan_to_num((np.sqrt((SD/N) - (M/N)**2)), nan=1.)

    return means, stds

def calc_z_score_at_once(config, data_dir, subject_id, metadata, available_channels, intervals, feature_dim):
    data = None
    for iv in intervals:
        start_ts, end_ts = iv
        inp_data = get_input_data(data_dir, subject_id, metadata, available_channels, start_ts, end_ts)
        if inp_data is None or len(inp_data) == 0:
            continue
        d = preprocess(config, inp_data)
        d = np.nan_to_num(d)
        if data is None:
            data = d
        else:
            data = np.vstack([data, d])
    return data.mean(axis=0), data.std(axis=0)


def normalize(data, means, stds):
    norm_data = (data - means) / stds
    return norm_data

def calc_z_score2(data):
    M = np.sum(data, axis=0)
    SD = np.sum(data**2, axis=0)
    N = data.shape[0]

    means = np.nan_to_num(M / N, nan=0.)
    stds = np.nan_to_num((np.sqrt(np.sum((data - M)**2) / N)), nan=1.)

    return means, stds


def add_random_noise(inp_data, freq, noise_factor=0.1):
    size, feature_dim = inp_data.shape
    d = np.array(inp_data, copy=True).swapaxes(0, 1).reshape(feature_dim, -1, freq)
    m = np.median(d, axis=2)
    r = np.random.uniform(low=0., high=1.-1e-12, size=m.shape)
    d = d + (m * r * noise_factor)[:, :, None]
    #d = d + (r * m).reshape(feature_dim, r.shape[1], 1).repeat(128, axis=2)
    d = d.reshape(feature_dim, size).swapaxes(0, 1)
    console_msg("HELLO")
    return d


def add_signal_based_noise(df):
    original_data = df.to_numpy()
    augmented_copy = np.empty_like(original_data)
    augmented_copy[:, 0] = original_data[:, 0]

    # BVP
    augmented_copy[:, 1] = smooth_magnitude_warping(original_data[:, 1], mu=1.0, sigma=0.35, noise_factor=1.0, cutoff=0.075, fs=128.0)
    # TEMP
    augmented_copy[:, 2] = smooth_additive_gaussian(original_data[:, 2], mu=0.0, sigma=1.0, noise_factor=0.05, cutoff=0.01, fs=128.0)

    # EDA, HR, ACCmag,x,y,z
    for ch in range(3, 9):
        augmented_copy[:, ch] = smooth_additive_gaussian(original_data[:, ch], mu=0.0, sigma=1.0, noise_factor=0.12, cutoff=0.075, fs=128.0)

    return pd.DataFrame(augmented_copy, columns=df.columns)


def smooth_magnitude_warping(signal, mu=1.0, sigma=0.35, noise_factor=0.1, cutoff=0.1, fs=128.0):
    random_factor = np.random.normal(mu, sigma, signal.shape)
    smooth_factor = smooth_lowpass_filter(random_factor, cutoff=cutoff, fs=fs)
    return signal * smooth_factor * noise_factor


def smooth_additive_gaussian(signal, mu=0.0, sigma=1.0, noise_factor=0.1, cutoff=0.1, fs=128.0):
    random_noise = np.random.normal(mu, sigma, signal.shape)
    smooth_noise = smooth_lowpass_filter(random_noise, cutoff=cutoff, fs=fs)
    median_value = np.median(signal)
    scaled_noise = smooth_noise * median_value * noise_factor
    noisy_signal = signal + scaled_noise
    return noisy_signal


def smooth_lowpass_filter(signal, cutoff=0.1, fs=128.0):
    # low-pass Butterworth filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal, method='gust')


def add_noise(inp_data, freq):
    a = np.array(inp_data, dtype=np.float32)
    for j in range(0, a.shape[0], freq):
        a[j:j + freq, 1:] = np.median(a[j:j + freq, 1:], axis=0) \
                                      * np.random.uniform(1e-3, 1. - 1e-3, 1)
    a = np.nan_to_num(a)
    return pd.DataFrame(a, columns=inp_data.columns)


def add_noise2(inp_data, freq):
    a = np.array(inp_data, dtype=np.float32)
    for j in range(0, a.shape[0], freq):
        data = a[j:j + freq, 1:]
        median = np.median(np.nan_to_num(data), axis=0)
        noise_range = median * 0.1
        noise = np.random.uniform(low=-noise_range, high=noise_range, size=data.shape)
        a[j:j + freq, 1:] =  data + noise
    a = np.nan_to_num(a)
    return pd.DataFrame(a, columns=inp_data.columns)


def add_noise3(inp_data, freq):
    offset = np.random.randint(1, 15) * 8
    a = np.array(inp_data, dtype=np.float32)
    a = np.concatenate([a[offset:], a[:offset]], axis=0)
    for j in range(0, a.shape[0], freq):
        d = a[j:j + freq, 1:]
        noise_ratio = np.random.uniform(0.1, 0.9, 1)
        a[j:j + freq, 1:] = (1. - noise_ratio) * d + noise_ratio * np.median(d, axis=0)
    a = np.nan_to_num(a)
    return pd.DataFrame(a, columns=inp_data.columns)


def add_noise4(inp_data, freq):
    a = np.array(inp_data, dtype=np.float32)
    for j in range(0, a.shape[0], freq):
        med =  np.median(a[j:j + freq, 1:], axis=0)
        rnd = np.random.uniform(0., 1. - 1e-5, 1)
        a[j:j + freq, 1:] += med * rnd
    a = np.nan_to_num(a)
    return pd.DataFrame(a, columns=inp_data.columns)

