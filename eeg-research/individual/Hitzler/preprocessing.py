import argparse
import os
from datetime import datetime

import mne.io
import numpy as np
from numpy import save

from utils import get_seizure_times


def process_data(data_dir, save_location, channels, alternative_channel_names):

    if not os.path.isdir(os.path.join(save_location, 'edf')):
        os.mkdir(os.path.join(save_location, 'edf'))
    if not os.path.isdir(os.path.join(save_location, 'labels')):
        os.mkdir(os.path.join(save_location, 'labels'))
    # get list of patients
    patients = os.listdir(data_dir)
    patients_path = [os.path.join(data_dir, f_) for f_ in patients]
    no_of_seizures = 0
    no_of_non_seizures = 0
    for iPatient in range(len(patients_path)):
        patient = patients_path[iPatient]
        # get list of sessions for patient
        sessions = os.listdir(patient)
        sessions_path = [os.path.join(patient, f_) for f_ in sessions]
        no_of_sessions = len(sessions_path)  # number of sessions
        for iSession in range(no_of_sessions):
            session = sessions_path[iSession]
            # get list of montage for session
            montage_list = os.listdir(session)
            montage_list_path = [os.path.join(session, f_) for f_ in montage_list]
            if len(montage_list_path) > 1:
                print('more than one montage found in session folder')
            for montage in montage_list_path:
                # get list of recordings for montage
                rec_edfs = [os.path.join(montage, f_) for f_ in os.listdir(montage) if f_.endswith('.edf')]
                # get list of csv_bi files for montage
                csv_bins = [os.path.join(montage, f_) for f_ in os.listdir(montage) if f_.endswith('.csv_bi')]
                for irec in range(len(rec_edfs)):
                    print('processing patient: %s, session: %s, recording: %s' % (patient, session, montage))
                    # read edf file
                    edfPath = os.path.join(montage, rec_edfs[irec])
                    eeg = mne.io.read_raw_edf(edfPath, preload=True)

                    # get sampling frequency
                    sfreq = eeg.info['sfreq']
                    if sfreq < 200:
                        print('Sampling frequency is less than 200Hz')
                        continue
                    # pick channels, if channel not found in recording, skip

                    if not set(channels).issubset(set(eeg.ch_names)):
                        print('Channels not found in recording, trying alternative channels')
                        if not set(alternative_channel_names).issubset(set(eeg.ch_names)):
                            print('Alternative channels not found in recording')
                            continue
                        else:
                            eeg.pick(alternative_channel_names)
                    else:
                        eeg.pick(channels)

                        # rename channels to standard names
                    # set measurement date to today if not set, otherwise error occurs
                    if eeg.info['meas_date'].year < 2000:
                        eeg.set_meas_date((datetime.today().timestamp(), 0))

                    # if length of recording is less than 30s, skip
                    if eeg.times[-1] < 30:
                        continue

                    mne.rename_channels(eeg.info, {ch: ch.replace('EEG ', '').replace('-LE', '').replace('-REF', '') for ch in eeg.info["ch_names"]})


                    # calculate bipolar montages
                    eeg = eeg.set_montage('standard_1020', match_case=False)


                    # Apply the bipolar reference using mne.set_bipolar_reference
                    anodes = ['FP1', 'F7', 'T3', 'T5', 'FP2', 'F8', 'T4', 'T6', 'T3', 'C3', 'CZ', 'C4', 'FP1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4']
                    cathodes = ['F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6', 'O2', 'C3', 'CZ', 'C4', 'T4', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2']
                    raw_bipolar = mne.set_bipolar_reference(eeg, anode=anodes, cathode=cathodes, drop_refs=True)
                    # drop PZ and FZ channels
                    raw_bipolar.drop_channels(['PZ', 'FZ'])

                    # create epochs of ~30s
                    epochs = mne.make_fixed_length_epochs(eeg, duration=30, preload=True)
                    # resample to 200Hz
                    epochs = epochs.resample(200)
                    epochs_data = epochs.get_data(copy=False)

                    bipolar_epochs = mne.make_fixed_length_epochs(raw_bipolar, duration=30, preload=True)
                    bipolar_epochs = bipolar_epochs.resample(200)
                    bipolar_epochs_data = bipolar_epochs.get_data(copy=False)

                    # read csv_bi file containing start and stop times of seizures
                    csvPath = os.path.join(montage, csv_bins[irec])
                    # get seizure times from csv file
                    seizureTimes = get_seizure_times(csvPath)

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

                    for i in range(len(labels_data_list)):
                        bipolar_epoch = bipolar_epochs_data[i]
                        bipolar_epoch = bipolar_epoch[np.newaxis, :, :]
                        patient_name = patient.split('/')[-1].split("\\")[-1]
                        saveFilename = 'DataArray_Patient_'+ str(patient_name)+ "_" + str(iPatient).zfill(3) + "_Session" + str(iSession).zfill(
                            3) + "_Rec" + str(irec).zfill(3) + "_Split" + str(i).zfill(3)
                        # save labels to labels folder
                        save(os.path.join(save_location, 'labels', saveFilename + ".labels"), labels_data_list[i])
                        # save epochs to edf folder
                        epochs[i].save(os.path.join(save_location, 'edf', saveFilename + "-unipolar-epo.fif"), overwrite=True)
                        # save bipolar epochs to edf folder
                        bipolar_epochs[i].save(os.path.join(save_location, 'edf', saveFilename + "-bipolar-epo.fif"), overwrite=True)
    print('Number of seizures: ', no_of_seizures)
    print('Number of non-seizures: ', no_of_non_seizures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw/dev')
    parser.add_argument('--save_location', type=str, default='data/processed/dev')
    parser.add_argument('--channels', type=str, default='EEG FP1-REF, EEG FP2-REF, EEG F3-REF, EEG F4-REF, EEG C3-REF, EEG C4-REF, EEG P3-REF, EEG P4-REF, EEG O1-REF, EEG O2-REF, EEG F7-REF, EEG F8-REF, EEG T3-REF, EEG T4-REF, EEG T5-REF, EEG T6-REF, EEG CZ-REF, EEG PZ-REF, EEG FZ-REF')

    # alternative channels
    parser.add_argument('--alternative_channel_names', type=str, default='EEG FP1-LE, EEG FP2-LE, EEG F3-LE, EEG F4-LE, EEG C3-LE, EEG C4-LE, EEG P3-LE, EEG P4-LE, EEG O1-LE, EEG O2-LE, EEG F7-LE, EEG F8-LE, EEG T3-LE, EEG T4-LE, EEG T5-LE, EEG T6-LE, EEG CZ-LE, EEG PZ-LE, EEG FZ-LE')
    args = parser.parse_args()
    data_dir = args.data_dir
    save_location = args.save_location
    channels = [channel.strip() for channel in args.channels.split(',')]
    # alternative channels
    alternative_channel_names = [channel.strip() for channel in args.alternative_channel_names.split(',')]
    process_data(data_dir, save_location, channels, alternative_channel_names)
