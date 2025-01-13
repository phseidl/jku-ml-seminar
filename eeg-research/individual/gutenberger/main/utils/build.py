import numpy as np
import math
import mne
import sys
import os


def segment_eeg(eeg, window_size=100, stride=50):
    """ Session EEG Signal by Slinding Window """
    n_chan, n_timep = eeg.shape
    tstamps, segments = [], []
    for i in range(0, n_timep, stride):
        seg = eeg[:,i: i + window_size]
        if seg.shape != (n_chan, window_size):
            break
        segments.append(seg)
        tstamps.append(i)

    return segments, tstamps

def create_dataset(x_fpath, y_fpath, fmt_terms, tmin, tmax, ch_names=None, win_size=4, stride=2):
    """ read mne set to numpy array """
    x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_terms[0]), verbose=0)
    sfreq = x_raw.info["sfreq"]
    win_size = math.ceil(win_size * sfreq)
    stride = math.ceil(stride * sfreq)
    nb_chan = len(x_raw.ch_names if ch_names is None else ch_names)
    tmin = math.ceil(tmin * sfreq)
    tmax = math.ceil(tmax * sfreq)

    X = np.zeros((0, nb_chan, win_size), dtype=np.float32)
    y = np.zeros((0, nb_chan, win_size), dtype=np.float32)
    for fmt_term in fmt_terms:
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_term), verbose=0)
        y_raw = mne.io.read_raw_eeglab(y_fpath.format(*fmt_term), verbose=0)
        if len(x_raw.ch_names) != len(y_raw.ch_names):
            raise ValueError(f"EEG channel should be matched, found {len(x_raw.ch_names)} and {len(y_raw.ch_names)}")

        x_content = x_raw[:, tmin: tmax][0]
        y_content = y_raw[:, tmin: tmax][0]
        if ch_names is not None:  # channel selection
            picks = [x_raw.ch_names.index(ch) for ch in ch_names]
            x_content = x_content[picks]
            y_content = y_content[picks]
        
        x_seg = np.array(segment_eeg(x_content, win_size, stride)[0])
        y_seg = np.array(segment_eeg(y_content, win_size, stride)[0])
        X = np.append(X, np.array(x_seg), axis=0)
        y = np.append(y, np.array(y_seg), axis=0)
    return X*1e+6, y*1e+6

def calc_SNR(clean_data, noisy_data, inDezibel = True):
    # clean data: reference data
    # noisy data: data to measure SNR on, e.g. output of the model
    n_chan = clean_data.shape[0]

    if inDezibel:
        return 1/n_chan * np.sum(10 * np.log10(np.linalg.norm(clean_data, axis = 1)/np.linalg.norm(clean_data-noisy_data, axis = 1)))
    else:
        return 1/n_chan * np.sum(np.linalg.norm(clean_data, axis = 1)**2/np.linalg.norm(clean_data-noisy_data, axis = 1)**2)

def create_EEG_DenoiseNet_dataset(config, artifact_type, debug = False):
    #TODO what about the upsampling?

    np.random.seed(0)  # for reproducibility
    snr_eog = np.linspace(-7,2, 10) # in dB
    snr_emg = np.linspace(-7,4, 12) # in dB

    Y = []
    X = []

    y = np.load(config['EEG_path'])        # clean segments
    
    
    #l_sum = 0

    ############### EOG ##################
    if artifact_type == 'EOG':
        n_EOG = np.load(config['EOG_path'])    #EOG noise segments
        nr_eog_segs = n_EOG.shape[0]
        selected_eeg_indices = np.random.choice(y.shape[0], nr_eog_segs, replace=False)
        selected_eeg_segments = y[selected_eeg_indices]

        #l = np.sum(x[0][:]**2)/np.sum(n_EOG[0][:]**2)/(10**(snr/5)) 
        #EOG: 3400 segments, EEG: 4514 segments
        #for i in range(nr_eog_segs):
            #l_sum += np.linalg.norm(selected_eeg_segments[i][:])/np.linalg.norm(n_EOG[i][:])/(10**(snr/10))
        #l = l_sum/nr_eog_segs
        for snr in snr_eog:
            l = np.linalg.norm(selected_eeg_segments.flatten())/np.linalg.norm(n_EOG.flatten())/(10**(snr/10))
            x = selected_eeg_segments + l*n_EOG
            if debug:
                snr_check_eog = calc_SNR(np.expand_dims(selected_eeg_segments.flatten(), 0), np.expand_dims(x.flatten(), 0))
            Y.append(selected_eeg_segments)
            X.append(x)

    elif artifact_type == 'EMG': 
    ############### EMG ##################
        n_EMG = np.load(config['EMG_path'])    #EMG noise segments
        nr_emg_segs = n_EMG.shape[0]
        np.random.shuffle(n_EMG)
        selected_eeg_indices = np.random.choice(y.shape[0], nr_emg_segs - y.shape[0], replace=False)
        selected_eeg_segments = y[selected_eeg_indices]
        y_expanded = np.vstack((y,selected_eeg_segments))
        for snr in snr_emg:
            l = np.linalg.norm(y_expanded.flatten())/np.linalg.norm(n_EMG.flatten())/(10**(snr/10))
            x = y_expanded + l*n_EMG
            if debug:
                snr_check_emg = calc_SNR(np.expand_dims(y_expanded.flatten(), 0), np.expand_dims(x.flatten(), 0))
            X.append(x)
            Y.append(y_expanded)
    else:
        raise Exception("Wrong artifact type.")
    
    X = np.vstack(X)
    Y = np.vstack(Y)
    
    return X, Y


def get_rdm_EEG_segment_DenoiseNet (config, artifact_type, snr, debug=False):
    np.random.seed(0)  # for reproducibility

    y = np.load(config['EEG_path'])        # clean segments
    
    #l_sum = 0

    ############### EOG ##################
    if artifact_type == 'EOG':
        n_EOG = np.load(config['EOG_path'])    #EOG noise segments
        random_idx_eeg = np.random.choice(y.shape[0], 1, replace=False)
        random_idx_noise = np.random.choice(n_EOG.shape[0], 1, replace=False)
        selected_clean_segment = y[random_idx_eeg]
        selected_noise_segment = n_EOG[random_idx_noise]

        l = np.linalg.norm(selected_clean_segment)/np.linalg.norm(selected_noise_segment)/(10**(snr/10))
        x = selected_clean_segment + l*selected_noise_segment
        if debug:
            snr_check_eog = calc_SNR(np.expand_dims(selected_clean_segment, 0), np.expand_dims(x, 0))

    elif artifact_type == 'EMG': 
    ############### EMG ##################
        n_EMG = np.load(config['EMG_path'])    #EMG noise segments
        random_idx_eeg = np.random.choice(y.shape[0], 1, replace=False)
        random_idx_noise = np.random.choice(n_EMG.shape[0], 1, replace=False)
        selected_clean_segment = y[random_idx_eeg]
        selected_noise_segment = n_EMG[random_idx_noise]

        l = np.linalg.norm(selected_clean_segment)/np.linalg.norm(selected_noise_segment)/(10**(snr/10))
        x = selected_clean_segment + l*selected_noise_segment
        if debug:
            snr_check_emg = calc_SNR(np.expand_dims(selected_clean_segment, 0), np.expand_dims(x, 0))
    
    else:
        raise Exception("Wrong artifact type.")
    
    return x, selected_clean_segment