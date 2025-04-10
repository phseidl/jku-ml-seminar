import numpy as np
import math
import mne
import warnings
import random

def generate_random_pairs(channel_nr, N, ch_dropout=None):
    """
    Generate N random pairs of channel indices without repetition, with a specified percentage of dropout.
    
    Args:
    - channel_nr (int): Total number of channels.
    - N (int): Number of pairs to generate.
    - ch_dropout (float): factor of channels to drop (0 to 1).
    
    Returns:
    - List of random pairs with dropout applied.
    """
    if ch_dropout is None:
        all_pairs = [(i, j) for i in range(channel_nr) for j in range(channel_nr) if i != j]   
    else:
        # Determine the number of channels to keep after dropout
        dropout_count = int(channel_nr * ch_dropout)
        remaining_channels = np.random.choice(channel_nr, channel_nr - dropout_count, replace=False)
        # Generate all possible pairs from the remaining channels
        all_pairs = [(i, j) for i in remaining_channels for j in remaining_channels if i != j]
    np.random.shuffle(all_pairs)
    
    return all_pairs[:N]


def create_montage(eeg_x, eeg_y, channel_order, montage = 'tuh', debug = False):
    channel_index_map = {name: idx for idx, name in enumerate(channel_order)}
    bipolar_data_x = []
    bipolar_data_y = []
    bipolar_names = []
    bipolar_positions = []
    if montage == 'tuh':
        montage_pairs = [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
        ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
        ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
        ('C4', 'P4'), ('P4', 'O2')
        ]

        # Create the bipolar pairs
        for anode, cathode in montage_pairs:
            try:
                anode_idx = channel_index_map[anode]
                cathode_idx = channel_index_map[cathode]
                bipolar_signal_x = eeg_x[anode_idx] - eeg_x[cathode_idx]
                bipolar_signal_y = eeg_y[anode_idx] - eeg_y[cathode_idx]
                bipolar_data_x.append(bipolar_signal_x)
                bipolar_data_y.append(bipolar_signal_y)
                bipolar_names.append(f"{anode}-{cathode}")
            except KeyError:
              if debug:
                  print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

        # Convert the bipolar data list to a numpy array
        bipolar_data_x = np.array(bipolar_data_x)
        bipolar_data_y = np.array(bipolar_data_y)

        if debug:
            # Output
            print("Bipolar Data Shape:", bipolar_data_x.shape)
            print("Bipolar Channels:", bipolar_names)
            print("Bipolar Data", bipolar_data_x)
            print("Noisy EEG segment", eeg_x)
    elif montage == 'tuh_rand':
        montage_pairs = [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
        ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
        ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
        ('C4', 'P4'), ('P4', 'O2')
        ]

        random.shuffle(montage_pairs)

        # Create the bipolar pairs
        for anode, cathode in montage_pairs:
            try:
                anode_idx = channel_index_map[anode]
                cathode_idx = channel_index_map[cathode]
                bipolar_signal_x = eeg_x[anode_idx] - eeg_x[cathode_idx]
                bipolar_signal_y = eeg_y[anode_idx] - eeg_y[cathode_idx]
                bipolar_data_x.append(bipolar_signal_x)
                bipolar_data_y.append(bipolar_signal_y)
                bipolar_names.append(f"{anode}-{cathode}")
            except KeyError:
              if debug:
                  print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

        # Convert the bipolar data list to a numpy array
        bipolar_data_x = np.array(bipolar_data_x)
        bipolar_data_y = np.array(bipolar_data_y)
    elif montage == 'random':
        montage_pairs = generate_random_pairs(len(channel_order), N=20)
        # Create the bipolar pairs
        for anode_idx, cathode_idx in montage_pairs:
            try:
                bipolar_signal_x = eeg_x[anode_idx] - eeg_x[cathode_idx]
                bipolar_signal_y = eeg_y[anode_idx] - eeg_y[cathode_idx]
                bipolar_data_x.append(bipolar_signal_x)
                bipolar_data_y.append(bipolar_signal_y)
                bipolar_names.append(f"{channel_order[anode_idx]}-{channel_order[cathode_idx]}")
            except KeyError:
                if debug:
                    print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")
        # Convert the bipolar data list to a numpy array
        bipolar_data_x = np.array(bipolar_data_x)
        bipolar_data_y = np.array(bipolar_data_y)
    elif montage == None or 'no_montage':
        bipolar_data_x = eeg_x
        bipolar_data_y = eeg_y
    return bipolar_data_x, bipolar_data_y

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

def create_dataset(x_fpath, y_fpath, fmt_terms, tmin, tmax, ch_names=None, win_size=4, stride=2, use_montage = 'tuh', downsample=False):
    """ read mne set to numpy array """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x_raw = mne.io.read_raw_eeglab(x_fpath.format(*fmt_terms[0]), verbose=0)
    if downsample:
        sfreq = x_raw.info["sfreq"]/4
    else:
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

        if downsample:
            #downsample to haf the freq
            x_content = x_content[:, ::4]
            y_content = y_content[:, ::4]

        #np.shape(x_content)
        #plt.hist(x_content*1e+6)
        #plt.show()
        x_content = x_content*1e+6
        y_content = y_content*1e+6

        if ch_names is not None:  # channel selection
            picks = [x_raw.ch_names.index(ch) for ch in ch_names]
            x_content = x_content[picks]
            y_content = y_content[picks]
        
        x_content, y_content = create_montage(x_content, y_content, x_raw.ch_names, montage = use_montage)   #if using the REF channels: montage='no_montage'

        #x_content, y_content = normalize_segments(x_content, y_content)
    
        x_seg = np.array(segment_eeg(x_content, win_size, stride)[0])
        y_seg = np.array(segment_eeg(y_content, win_size, stride)[0])

        x_seg, y_seg = normalize_segments(x_seg, y_seg)

        X = np.append(X, np.array(x_seg), axis=0)
        y = np.append(y, np.array(y_seg), axis=0)
        print(f'Created dataset for subject {fmt_term}')
        
    return X, y   
    
def normalize_segments(eeg_data_x, eeg_data_y, debug=False):
    #print(f'EEG_data_x.shape: {eeg_data_x.shape}')
    #eeg_data.shape: (sample_nr, ch, seq_len)
    #eeg_data_x = np.transpose(eeg_data_x, (2,1,0))
    #eeg_data_y = np.transpose(eeg_data_y, (2,1,0))
    # Calculate the 95th percentile of the absolute value for each channel in the segment
    percentiles = np.percentile(np.abs(eeg_data_x), 95, axis=-1, keepdims=True)

    if debug:
        # Check for zero values in the percentiles
        if np.any(percentiles == 0):
            print("Zero value detected in percentiles:")
            print("Percentiles:", percentiles)
            print("EEG Data (eeg_data_x):", eeg_data_x)
            print("EEG Data (eeg_data_y):", eeg_data_y)

    #percentiles = np.where(percentiles == 0, 1, percentiles) #avoid zero values by replacing with 1e-7

    # Normalize each EEG segment by dividing by the 95th percentile of the absolute value of each channel
    x_norm = eeg_data_x / percentiles
    y_norm = eeg_data_y / percentiles
    # Identify segments without NaN values
    mask = ~np.isnan(x_norm).any(axis=(1, 2))    #shape (nr of segments,)

    if not mask.all():  # `mask.all()` is True if all values are True
        print("Some segments contain NaN values and will be removed.") 
    
    # Filter out segments with NaN values
    x_norm = x_norm[mask]
    y_norm = y_norm[mask]
    
    return x_norm, y_norm


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