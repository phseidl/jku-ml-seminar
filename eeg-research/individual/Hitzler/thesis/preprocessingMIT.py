import pickle
import re

from mne import EpochsArray

from utils import get_seizure_times_mit

import argparse
import os

from datetime import datetime

import mne.io
import numpy as np
from numpy import save



def process_data(data_dir, save_location, channels):
    if not os.path.isdir(os.path.join(save_location, 'edf')):
        os.mkdir(os.path.join(save_location, 'edf'))
    if not os.path.isdir(os.path.join(save_location, 'labels')):
        os.mkdir(os.path.join(save_location, 'labels'))
    # get list of patients
    sessions = os.listdir(data_dir)
    sessions_path = [os.path.join(data_dir, f_) for f_ in sessions]
    no_of_seizures = 0
    no_of_non_seizures = 0
    f = open("channel_errors.txt", "a")
    seizure_lengths = []
    for iSession in range(len(sessions_path)):
        summary = open(os.path.join(sessions_path[iSession], sessions[iSession]+"-summary.txt")).readlines()
        session_path = sessions_path[iSession]
        session = sessions[iSession]
        # get list of recordings for montage
        rec_edfs = [os.path.join(session_path, f_) for f_ in os.listdir(session_path) if f_.endswith('.edf')]
        # get list of csv_bi files for montage
        csv_bins = [os.path.join(session_path, f_) for f_ in os.listdir(session_path) if f_.endswith('.edf.seizures')]
        for irec in range(len(rec_edfs)):
            print('processing session: %s, recording: %s' % (session_path, rec_edfs[irec]))
            # read edf file
            edfPath = os.path.join(rec_edfs[irec])
            eeg = mne.io.read_raw_edf(edfPath, preload=True)
            # drop dummy channels
            try:
                eeg.pick(channels)
            except Exception as e:
                print(e)
                f.write(str(str(e) + "\n"))
                print("channels not present")
                continue

            # get sampling frequency
            sfreq = eeg.info['sfreq']
            if sfreq < 200:
                print('Sampling frequency is less than 200Hz')
                continue
            # pick channels, if channel not found in recording, skip

                # rename channels to standard names
            # set measurement date to today if not set, otherwise error occurs
            if eeg.info['meas_date'].year < 2000 or eeg.info['meas_date'].year > 2030:
                eeg.set_meas_date((datetime.today().timestamp(), 0))

            # if length of recording is less than 30s, skip
            if eeg.times[-1] < 30:
                continue

            # create epochs of ~30s
            epochs = mne.make_fixed_length_epochs(eeg, duration=30, preload=True)
            # resample to 200Hz
            epochs = epochs.resample(200)
            epochs_data = epochs.get_data(copy=False)

            # add four additional channels with zeros at index 8,9,10,11
            epochs_data = np.concatenate((epochs_data, np.zeros((epochs_data.shape[0], 4, epochs_data.shape[2]))), axis=1)
            # move them to index 8,9,10,11
            epochs_data = np.concatenate((epochs_data[:, :8, :], epochs_data[:, 16:, :], epochs_data[:, 8:16, :]), axis=1)
            # create new metadata
            old_ch_names = epochs.ch_names
            new_ch_names = old_ch_names[:8] + ['dummy1', 'dummy2', 'dummy3', 'dummy4'] + old_ch_names[8:]
            new_info = mne.create_info(
                ch_names=new_ch_names,
                sfreq=epochs.info['sfreq']
            )
            epochs = EpochsArray(epochs_data, new_info, events=epochs.events, event_id=epochs.event_id,
                         tmin=epochs.tmin, baseline=epochs.baseline)
            # find seizure times in summary file
            seizureTimes = get_seizure_times_mit(summary, os.path.basename(rec_edfs[irec]))
            [seizure_lengths.append(entry[1] - entry[0]) for entry in seizureTimes]

            # create label vector, 1 = seizure, 0 = no seizure, length = number of epochs * window_size * sample_rate
            labels = np.zeros(epochs_data.shape[0] * epochs_data.shape[2])

            window_size = 30
            sample_rate = 200

            for i in range(len(seizureTimes)):
                # get start and stop indices of seizure
                startIdx = int(seizureTimes[i][0] * sample_rate)
                stopIdx = int(seizureTimes[i][1] * sample_rate)
                labels[startIdx:stopIdx] = 1
            labels_data_list = []

            interval_size = int(window_size * sample_rate)

            # Calculate the total number of intervals
            num_intervals = epochs_data.shape[0]
            # split data into intervals of 30s
            for i in range(num_intervals):
                start_index = i * interval_size
                end_index = (i + 1) * interval_size
                # check if interval contains seizure or not for statistics
                if labels[start_index:end_index].sum() == 0:
                    no_of_non_seizures += 1
                elif labels[start_index:end_index].sum() > 0:
                    no_of_seizures += 1
                labels_data_list.append(labels[start_index:end_index])

            # calculate STFT for each interval and save each epoch + labels as a separate file
            for i in range(len(labels_data_list)):
                saveFilename = 'DataArray_' + str(session) + "_Rec" + str(irec).zfill(3) + "_Split" + str(i).zfill(3)
                # save labels to labels folder
                save(os.path.join(save_location, 'labels', saveFilename + ".labels"), labels_data_list[i])
                # save epochs to edf folder
                epochs[i].save(os.path.join(save_location, 'edf', saveFilename + "-epo.fif"), overwrite=True)

    f.close()
    print('Number of seizures: ', no_of_seizures)
    print('Number of non-seizures: ', no_of_non_seizures)
    print('Seizure average length: ', {(sum(seizure_lengths) / len(seizure_lengths))})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mit/train')
    parser.add_argument('--save_location', type=str, default='data/mit/processed/train')
    parser.add_argument('--channels', type=str,
                        default='FP1-F7, F7-T7, T7-P7, P7-O1, FP2-F8, F8-T8, T8-P8-0, P8-O2, P7-T7, T7-FT9, C3-P3, P3-O1, FT9-FT10, FT10-T8, C4-P4, P4-O2')
    args = parser.parse_args()
    data_dir = '/mnt/cache/data/MIT/raw/all'
    save_location = '/mnt/cache/data/MIT/processed/all'
    channels = [channel.strip() for channel in args.channels.split(',')]

    process_data(data_dir, save_location, channels)
