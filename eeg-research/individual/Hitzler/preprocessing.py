import os
from datetime import datetime

import mne.io
import numpy as np
from numpy import save

from individual.Hitzler.utils import get_seizure_times


def process_data(data_dir, save_location, channels):

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
                    try:
                        eeg.pick(channels)
                    except:
                        channels_2 = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG FZ-LE']
                        if not set(channels_2).issubset(set(eeg.ch_names)):
                            print('Channel not found in recording')
                            continue
                        eeg.pick(channels_2)

                    # set measurement date to today if not set, otherwise error occurs
                    if eeg.info['meas_date'].year < 2000:
                        eeg.set_meas_date((datetime.today().timestamp(), 0))

                    # if length of recording is less than 30s, skip
                    if eeg.times[-1] < 30:
                        continue
                    # create epochs of ~30s
                    epochs = mne.make_fixed_length_epochs(eeg, duration=30, preload=True)
                    # resample to 200Hz
                    epochs = epochs.resample(200)
                    epochs_data = epochs.get_data(copy=False)

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
                        if labels[start_index:end_index].sum() == 0:
                            no_of_non_seizures += 1
                        elif labels[start_index:end_index].sum() > 0:
                            no_of_seizures += 1
                        labels_data_list.append(labels[start_index:end_index])


                    for i in range(len(labels_data_list)):
                        patient_name = patient.split('/')[-1].split("\\")[-1]
                        saveFilename = 'DataArray_Patient_'+ str(patient_name)+ "_" + str(iPatient).zfill(3) + "_Session" + str(iSession).zfill(
                            3) + "_Rec" + str(irec).zfill(3) + "_Split" + str(i).zfill(3)
                        # save labels to labels folder
                        save(os.path.join(save_location, 'labels', saveFilename + ".labels"), labels_data_list[i])
                        # save epochs to edf folder
                        epochs[i].save(os.path.join(save_location, 'edf', saveFilename + "-epo.fif"), overwrite=True)
    print('Number of seizures: ', no_of_seizures)
    print('Number of non-seizures: ', no_of_non_seizures)


# main
if __name__ == '__main__':
    data_dir = 'C:/Users/FlorianHitzler/Documents/Uni/Bachelor Thesis/new_download/jku-ml-seminar23/eeg-research/individual/Hitzler/data/raw/dev'
    save_location = 'C:/Users/FlorianHitzler/Documents/Uni/Bachelor Thesis/new_download/jku-ml-seminar23/eeg-research/individual/Hitzler/data/processed/dev'
    channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF','EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG FZ-REF']
    process_data(data_dir, save_location, channels)
