import mne 
import numpy as np
import os
import pickle
from tqdm import tqdm
import pandas as pd


def BuildEvents(signals, times, EventData):
    EventData = EventData.to_numpy()
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .csv file
    fs = 250.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.empty([numEvents, 1], dtype=str)  # channel that had the detected seizure
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]

    signals = np.concatenate([signals, signals, signals], axis=1)
    j = 0
    for i in range(numEvents):  # for each event
        chan = EventData[i, 0]  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end_indices = np.where((times) >= EventData[i, 2])[0]
        if end_indices.size > 0:
            # If the array is not empty, get the first index
            end = end_indices[0]
        else:
            # If the array is empty, set end to the last index of the times array
            end = len(times) - 1
        # Calculate the start and end indices of the slice
        start_index = offset + start - 2 * int(fs)
        end_index = offset + end + 2 * int(fs)
        # Limit the end index to the size of features[i, :]
        end_index = min(end_index, start_index + features.shape[2])
        # Check if the duration of the event is shorter than the size of the third dimension of features
        if end - start < features.shape[2]:
            # If it is, skip the current iteration of the loop
            continue
        features[j, :, :end_index-start_index] = signals[:, start_index:end_index]
        offending_channel[j, :] = chan
        labels[j, :] = int(EventData[i, 3])
        j += 1

    # Trim the arrays to the number of processed events
    features = features[:j]
    offending_channel = offending_channel[:j]
    labels = labels[:j]

    return [features, offending_channel, labels]  # return seizure types


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["FP1"]]
            - signals[signal_names["F7"]],  # 0
            (
                signals[signal_names["F7"]]
                - signals[signal_names["T3"]]
            ),  # 1
            (
                signals[signal_names["T3"]]
                - signals[signal_names["T5"]]
            ),  # 2
            (
                signals[signal_names["T5"]]
                - signals[signal_names["O1"]]
            ),  # 3
            (
                signals[signal_names["FP2"]]
                - signals[signal_names["F8"]]
            ),  # 4
            (
                signals[signal_names["F8"]]
                - signals[signal_names["T4"]]
            ),  # 5
            (
                signals[signal_names["T4"]]
                - signals[signal_names["T6"]]
            ),  # 6
            (
                signals[signal_names["T6"]]
                - signals[signal_names["O2"]]
            ),  # 7
            (
                signals[signal_names["FP1"]]
                - signals[signal_names["F3"]]
            ),  # 14
            (
                signals[signal_names["F3"]]
                - signals[signal_names["C3"]]
            ),  # 15
            (
                signals[signal_names["C3"]]
                - signals[signal_names["P3"]]
            ),  # 16
            (
                signals[signal_names["P3"]]
                - signals[signal_names["O1"]]
            ),  # 17
            (
                signals[signal_names["FP2"]]
                - signals[signal_names["F4"]]
            ),  # 18
            (
                signals[signal_names["F4"]]
                - signals[signal_names["C4"]]
            ),  # 19
            (
                signals[signal_names["C4"]]
                - signals[signal_names["P4"]]
            ),  # 20
            (signals[signal_names["P4"]] - signals[signal_names["O2"]]),
        )
    )  # 21
    return new_signals

# Define the seizure types
seizure_types = {
    "null": 0, "spsw": 1, "gped": 2, "pled": 3, "eybl": 4, "artf": 5, "bckg": 6, "seiz": 7, 
    "fnsz": 8, "gnsz": 9, "spsz": 10, "cpsz": 11, "absz": 12, "tnsz": 13, "cnsz": 14, 
    "tcsz": 15, "atsz": 16, "mysz": 17, "nesz": 18, "intr": 19, "slow": 20, "eyem": 21, 
    "chew": 22, "shiv": 23, "musc": 24, "elpp": 25, "elst": 26, "calb": 27, "hphs": 28, 
    "trip": 29, "elec": 30, "eyem_chew": 100, "eyem_shiv": 101, "eyem_musc": 102, 
    "eyem_elec": 103, "chew_shiv": 104, "chew_musc": 105, "chew_elec": 106, "shiv_musc": 107, 
    "shiv_elec": 108, "musc_elec": 109
}

def readEDFandCSV(fileName):
    # Read the .edf file with the EEG data
    Rawdata = mne.io.read_raw_edf(fileName)
    signals, times = Rawdata[:]

    # Rename the channels to remove 'EEG ', '-LE' and '-REF'
    # This way, they follow the same nomenclature as in the .csv file
    for i in range(len(Rawdata.ch_names)):
        newChLabel = Rawdata.ch_names[i].replace('EEG ', '').replace('-LE', '').replace('-REF', '')
        Rawdata.rename_channels({Rawdata.ch_names[i]: newChLabel})

    # Read the .csv file with the seizure data
    CsvFile = fileName[0:-3] + "csv"
    eventData = pd.read_csv(CsvFile, skiprows=5) # remove header
    # Map the seizure types to integers
    eventData['label'] = eventData['label'].map(seizure_types)
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    # Use os.path.join() to construct the filename
                    [signals, times, event, Rawdata] = readEDFandCSV(
                        os.path.join(dirName, fname)
                    )  # seizure label is the .csv file
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("Couldn't load " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUSZ dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""

root = "C:/Users/riza_/AppData/Roaming/MobaXterm/home/TUSZ-2.0.1/edf/"
train_out_dir = os.path.join(root, "processed_train")
eval_out_dir = os.path.join(root, "processed_eval")
dev_out_dir = os.path.join(root, "processed_dev")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(dev_out_dir):
    os.makedirs(dev_out_dir)
BaseDirTrain = os.path.join(root, "train")
fs = 250
TrainFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 16 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
)
BaseDirEval = os.path.join(root, "eval")
fs = 250
EvalFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 16 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
)
BaseDirDev = os.path.join(root, "dev")
fs = 250
DevFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 16 for channels, fs for num of points
DevLabels = np.empty([0, 1])
DevOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirDev, DevFeatures, DevLabels, DevOffendingChannel, dev_out_dir
)
