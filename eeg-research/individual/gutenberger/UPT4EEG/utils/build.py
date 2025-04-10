import numpy as np
import math
import mne
import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_terms[0]), verbose=0)
    sfreq = x_raw.info["sfreq"]
    win_size = math.ceil(win_size * sfreq)
    stride = math.ceil(stride * sfreq)
    nb_chan = len(x_raw.ch_names if ch_names is None else ch_names)
    tmin = math.ceil(tmin * sfreq)
    if tmax != 'max':
        tmax = math.ceil(tmax * sfreq)

    X = np.zeros((0, nb_chan, win_size), dtype=np.float32)
    y = np.zeros((0, nb_chan, win_size), dtype=np.float32)
    for fmt_term in fmt_terms:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_term), verbose=0)
            y_raw = mne.io.read_raw_eeglab(y_fpath.format(*fmt_term), verbose=0)
        if len(x_raw.ch_names) != len(y_raw.ch_names):
            raise ValueError(f"EEG channel should be matched, found {len(x_raw.ch_names)} and {len(y_raw.ch_names)}")

        if tmax == 'max':
            x_content = x_raw[:, tmin: x_raw.n_times-tmin][0]
            y_content = y_raw[:, tmin: x_raw.n_times-tmin][0]
        else:
            x_content = x_raw[:, tmin: tmax][0]
            y_content = y_raw[:, tmin: tmax][0]

        #np.shape(x_content)
        #plt.hist(x_content*1e+6)
        #plt.show()

        if ch_names is not None:  # channel selection
            picks = [x_raw.ch_names.index(ch) for ch in ch_names]
            x_content = x_content[picks]
            y_content = y_content[picks]

        x_seg = np.array(segment_eeg(x_content, win_size, stride)[0])
        y_seg = np.array(segment_eeg(y_content, win_size, stride)[0])
        X = np.append(X, np.array(x_seg), axis=0)
        y = np.append(y, np.array(y_seg), axis=0)
        #X, y = normalize_segments(X*1e+6, y*1e+6)
        print(f'Created dataset for subject {fmt_term}')
    X, y = remove_troubling_segments(X*1e+6, y*1e+6)
    return X, y, x_raw.ch_names    

def remove_troubling_segments(eeg_data_x, eeg_data_y, threshold=1e-3):
    # EEG data shape (nr of segments, nr of channels, seq len)
    # Calculate the 95th percentile of the absolute value for each channel in each segment
    percentile_95 = np.percentile(np.abs(eeg_data_x), 95, axis=2, keepdims=True)

    segments_to_keep = (percentile_95 >= threshold).all(axis=1).squeeze()  # Shape: (nr of segments,)

    return eeg_data_x[segments_to_keep], eeg_data_y[segments_to_keep]