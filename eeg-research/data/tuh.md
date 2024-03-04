# Instructions to obtain TUH dataset

- 33 channels @ 250 Hz for 20 min per edf; LowPass at 125.00 Hz

## Access for people using the JKU servers:

TUHEV can be found at: ```/system/user/publicwork/eeg/data/TUH/```

If you need access to the JKU servers - write me a mail.

## Get access to the dataset
To request access to the TUH EEG Corpus, please fill out [this form](https://isip.piconepress.com/projects/tuh_eeg/html/data_sharing.pdf), follow the instructions given, and email a signed copy to help@nedcdata.org. Please include "Download The TUH EEG Corpus" in the subject line.

In order to prevent issues with the "Your Personal Post Office Address At The Above Institution" field of the form, this is an address that works:

\~Your Name~ \
Johannes Kepler University \
Institute for Machine Learning \
Altenbergerstraße 69 \
4040 Linz \
Austria

## Downalod the data
### For the TUSZ - TUH EEG Seizure Corpus, using rsync:
```
rsync -auxvL --delete nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/v2.0.1/ ./data/datasets/TUSZ/
```
### For the TUEV - TUH EEG Events Corpus:
EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).
```
rsync -auxvL --delete nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/v2.0.0/ ./data/datasets/TUEV/
```

The username and password should have been sent to you by mail, after following the instructions above.

## Add/Modify dataset config

## Annotation description
can be found [here](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fisip.piconepress.com%2Fpublications%2Freports%2F2020%2Ftuh_eeg%2Fannotations%2Fannotation_guidelines_v39.docx&wdOrigin=BROWSELINK)
