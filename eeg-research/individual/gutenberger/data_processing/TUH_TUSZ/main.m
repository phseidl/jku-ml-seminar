startsub = 3;
stopsub = 15;
root = "edf/dev";    %root directory of TUH dataset, e.g. 'C:/.../edf/dev'
montage = "02_tcp_le";
saveroot = 'TUH_dataset_PROCESSED';  
chanLocsPath = 'utils/Standard-10-20-Cap81-EDIT.ced';
channels = {'FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'};
removeSeiz = true;
ica = true;
saveArrays = false;

process_TUH_data(startsub,stopsub,montage,root,saveroot, chanLocsPath, channels, removeSeiz, ica, saveArrays)

               